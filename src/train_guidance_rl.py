"""
RL fine-tuning of the visual guidance policy using REINFORCE.

The policy is initialized from a supervised checkpoint, then trained to
maximize expected reward by sampling candidate boxes and updating via
policy gradient.

Two reward modes:
  "grounding"  (fast)  : IoU + format_reward - area_penalty  (no VLM call)
  "combined"   (slow)  : grounding + answer_reward            (VLM call per sample)

Policy: Gaussian over raw box parameters (cx, cy, w, h before sigmoid).
  action ~ N(mean, std * I)
  log_prob = sum_i log N(action_i; mean_i, std_i)
  loss = -log_prob * advantage
  advantage = reward - baseline  (moving average baseline)

CLI:
    # Fast grounding-only RL:
    python src/train_guidance_rl.py \\
        --data data/vsr_guidance.json \\
        --sft-checkpoint ../checkpoints/guidance_sft/best.pt \\
        --reward-type grounding \\
        --epochs 5 --batch-size 16 --samples 4

    # Slow answer-aware RL (requires GPU, VLM calls):
    python src/train_guidance_rl.py \\
        --data data/vsr_guidance.json \\
        --sft-checkpoint ../checkpoints/guidance_sft/best.pt \\
        --reward-type combined \\
        --model qwen \\
        --epochs 3 --batch-size 8 --samples 2

Outputs:
    ../checkpoints/guidance_rl/best.pt
    ../checkpoints/guidance_rl/final.pt
    ../checkpoints/guidance_rl/rl_log.json
    ../checkpoints/guidance_rl/config.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guidance_dataset import VSRGuidanceDataset, _make_splits
from guidance_model import GuidancePolicy, collate_fn, raw_to_box
from guidance_rewards import compute_rewards_batch
from train_guidance_sft import _box_iou_scalar, validate, make_collate
from torch.utils.data import DataLoader

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Log-std range for numerical stability
LOG_STD_MIN = -4.0
LOG_STD_MAX = 0.5


# ---------------------------------------------------------------------------
# Gaussian policy wrapper
# ---------------------------------------------------------------------------

class GaussianGuidancePolicy(nn.Module):
    """
    Wraps GuidancePolicy with a learned log-std parameter for REINFORCE.

    The base policy (CLIP + BoxHead) outputs the mean of the Gaussian.
    A separate learned vector provides per-dimension log-std.
    """

    def __init__(self, base: GuidancePolicy):
        super().__init__()
        self.base = base
        # Per-dimension log-std, initialized to log(0.1) ≈ -2.3
        self.log_std = nn.Parameter(torch.full((4,), -2.3))

    def forward_mean_raw(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw (pre-sigmoid) mean parameters (B, 4)."""
        img_emb  = self.base.encode_images(pixel_values)
        txt_emb  = self.base.encode_texts(input_ids, attention_mask)
        combined = torch.cat([img_emb, txt_emb], dim=-1)
        return self.base.box_head(combined)  # (B, 4) raw

    def sample(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample n_samples box proposals per example.

        Returns:
            raw_mean   : (B, 4) mean of the Gaussian
            raw_samples: (B, n_samples, 4) sampled raw values
            log_probs  : (B, n_samples) log-probabilities of samples
        """
        B = pixel_values.shape[0]
        raw_mean = self.forward_mean_raw(pixel_values, input_ids, attention_mask)  # (B, 4)

        std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX).exp()  # (4,)

        # Sample: (B, n_samples, 4)
        eps = torch.randn(B, n_samples, 4, device=raw_mean.device)
        raw_samples = raw_mean.unsqueeze(1) + std.unsqueeze(0).unsqueeze(0) * eps

        # log_prob of each sample under N(mean, std): sum over dims
        diff = (raw_samples - raw_mean.unsqueeze(1)) / (std + 1e-8)
        log_probs = -0.5 * (diff ** 2 + 2 * (std + 1e-8).log() + torch.log(torch.tensor(2 * 3.14159265))).sum(-1)

        return raw_mean, raw_samples, log_probs

    def save_checkpoint(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "box_head_state_dict": self.base.box_head.state_dict(),
            "log_std": self.log_std.data,
            "clip_model_name": self.base.processor.name_or_path,
            "meta": meta or {},
        }, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: Optional[torch.device] = None,
    ) -> "GaussianGuidancePolicy":
        ckpt = torch.load(path, map_location=device or "cpu")
        base = GuidancePolicy(clip_model_name=ckpt.get("clip_model_name", "openai/clip-vit-base-patch32"))
        base.box_head.load_state_dict(ckpt["box_head_state_dict"])
        policy = cls(base)
        if "log_std" in ckpt:
            policy.log_std.data = ckpt["log_std"]
        if device is not None:
            policy = policy.to(device)
        return policy


# ---------------------------------------------------------------------------
# VLM answering helper (for "combined" reward mode)
# ---------------------------------------------------------------------------

def _load_pil(path: Optional[str]):
    """Load a PIL image from path, returning None if unavailable."""
    if path is None or not os.path.exists(path):
        return None
    try:
        from PIL import Image
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _load_vlm(model_name: str):
    """Load VLM interface. Called once before the training loop."""
    from model_interface import QwenVLLocalInterface
    if model_name == "qwen":
        return QwenVLLocalInterface()
    raise ValueError(f"Unknown model for combined RL: {model_name!r}")


def get_vlm_answers(
    vlm,
    images,
    captions: list[str],
    boxes: list[list],
    cache: dict,
    prompt_template: str,
    full_crop: bool = True,
) -> list[Optional[str]]:
    """
    Query the VLM for each (image, box) pair and return parsed answers.

    When full_crop=True (default), sends [full_image, crop] together so the
    model can see both the spatial context and the highlighted region.
    When full_crop=False, sends only the crop.

    Caches results by (caption_prefix, box) to avoid redundant calls.
    """
    from crop_utils import safe_crop
    from parse_outputs import extract_json_object, parse_answer

    FULL_CROP_PROMPT = (
        "You are given two images: the full scene and a cropped region of interest.\n"
        "Use both to answer whether the following statement is true or false.\n"
        'Statement: "{caption}"\n'
        'Respond with JSON only: {"reasoning": "...", "answer": "true" or "false"}'
    )

    answers = []
    for img, caption, box in zip(images, captions, boxes):
        key = f"{caption[:40]}|fc={full_crop}|{[round(v, 3) for v in box] if box else 'none'}"
        if key in cache:
            answers.append(cache[key])
            continue

        if img is None:
            cache[key] = None
            answers.append(None)
            continue

        crop = safe_crop(img, box) if box is not None else img
        try:
            if full_crop and box is not None:
                prompt = FULL_CROP_PROMPT.replace("{caption}", caption)
                raw = vlm.generate_response_multi([img, crop], prompt)
            else:
                prompt = prompt_template.replace("{caption}", caption)
                raw = vlm.generate_response(crop, prompt)
            ans = parse_answer(extract_json_object(raw))
        except Exception:
            ans = None
        cache[key] = ans
        answers.append(ans)

    return answers


# ---------------------------------------------------------------------------
# Moving average baseline
# ---------------------------------------------------------------------------

class MovingAverageBaseline:
    def __init__(self, momentum: float = 0.99):
        self.value    = 0.0
        self.momentum = momentum
        self._initialized = False

    def update(self, reward: float) -> None:
        if not self._initialized:
            self.value = reward
            self._initialized = True
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * reward

    def __call__(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# RL training loop
# ---------------------------------------------------------------------------

def train_rl(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Reward mode: {args.reward_type}")

    os.makedirs(args.output_dir, exist_ok=True)
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load SFT checkpoint into Gaussian policy
    print(f"Loading SFT checkpoint: {args.sft_checkpoint}")
    policy = GaussianGuidancePolicy.load_checkpoint(args.sft_checkpoint, device=device)
    processor = policy.base.processor

    # Dataset
    cache_abs = args.data if os.path.isabs(args.data) else os.path.join(_ROOT, args.data)
    all_ds = VSRGuidanceDataset.from_cache(cache_abs, split="all", seed=args.seed)
    splits = _make_splits(all_ds.examples, args.seed)
    train_examples = splits["train"]
    # Limit training examples for combined mode (VLM calls are expensive)
    if args.max_train and len(train_examples) > args.max_train:  # noqa: E501
        random.seed(args.seed)
        train_examples = random.sample(train_examples, args.max_train)
        print(f"Combined RL: limiting to {args.max_train} training examples")
    train_ds = VSRGuidanceDataset(train_examples)
    val_ds   = VSRGuidanceDataset(splits["val"])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    collate = make_collate(processor, device)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate,
    )

    # Only update BoxHead + log_std
    params = list(policy.base.box_head.parameters()) + [policy.log_std]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    baseline = MovingAverageBaseline(momentum=0.95)
    vlm_cache: dict = {}
    best_iou = -1.0
    rl_log = []

    # Load VLM and prompt for combined mode (done once, before the epoch loop)
    vlm = None
    vlm_prompt = None
    if args.reward_type == "combined":
        print(f"Loading VLM ({args.model}) for combined reward...")
        vlm = _load_vlm(args.model)
        try:
            from utils import load_prompt_template
            vlm_prompt = load_prompt_template("textual_cot")
        except Exception:
            vlm_prompt = 'Answer true or false: {caption}\nRespond with JSON: {"answer": "true" or "false"}'
        print("VLM loaded.")

    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_rewards, epoch_ious = [], []
        total_pg_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            if batch is None:
                continue

            B = batch["pixel_values"].shape[0]
            target_boxes  = batch["target_boxes"]           # (B, 4)
            ground_truths = batch["answers"]

            # Sample K boxes per example
            raw_mean, raw_samples, log_probs = policy.sample(
                batch["pixel_values"],
                batch["input_ids"],
                batch["attention_mask"],
                n_samples=args.samples,
            )
            # raw_samples: (B, K, 4)  log_probs: (B, K)

            # Convert all samples to valid boxes
            B_K = B * args.samples
            raw_flat    = raw_samples.view(B_K, 4)
            boxes_flat  = raw_to_box(raw_flat)              # (B*K, 4)

            pred_list    = boxes_flat.detach().cpu().tolist()
            target_list  = target_boxes.repeat_interleave(args.samples, dim=0).cpu().tolist()
            gt_list      = [gt for gt in ground_truths for _ in range(args.samples)]

            # VLM answers (combined mode only)
            vlm_answers = None
            if args.reward_type == "combined":
                # Load PIL images from paths for each example, repeated K times
                pil_images_flat = []
                for i in range(B_K):
                    path = batch["image_paths"][i // args.samples]
                    img = _load_pil(path)
                    pil_images_flat.append(img)
                captions_flat = [batch["captions"][i // args.samples] for i in range(B_K)]
                vlm_answers = get_vlm_answers(
                    vlm             = vlm,
                    images          = pil_images_flat,
                    captions        = captions_flat,
                    boxes           = pred_list,
                    cache           = vlm_cache,
                    prompt_template = vlm_prompt,
                )

            rewards = compute_rewards_batch(
                pred_boxes       = pred_list,
                target_boxes     = target_list,
                ground_truths    = gt_list,
                predicted_answers = vlm_answers,
                reward_type      = args.reward_type,
            )
            reward_vals = torch.tensor(
                [r["total"] for r in rewards], dtype=torch.float32, device=device
            ).view(B, args.samples)  # (B, K)

            # Update baseline with mean reward
            mean_reward = reward_vals.mean().item()
            baseline.update(mean_reward)

            # Advantage
            advantage = (reward_vals - baseline()).detach()  # (B, K)

            # REINFORCE loss: -log_prob * advantage, averaged over B and K
            pg_loss = -(log_probs * advantage).mean()

            optimizer.zero_grad()
            pg_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_pg_loss += pg_loss.item()
            epoch_rewards.append(mean_reward)
            # Track mean IoU of best sample per example
            iou_per_sample = torch.tensor(
                [_box_iou_scalar(pred_list[i], target_list[i]) for i in range(B_K)]
            ).view(B, args.samples)
            epoch_ious.append(iou_per_sample.max(dim=1).values.mean().item())
            n_batches += 1

        # Validate using greedy (mean) box
        val_metrics = _validate_greedy(policy, val_ds, processor, device, args.batch_size)
        elapsed = time.time() - t0

        log_entry = {
            "epoch":         epoch,
            "mean_reward":   sum(epoch_rewards) / max(len(epoch_rewards), 1),
            "mean_pg_loss":  total_pg_loss / max(n_batches, 1),
            "train_iou":     sum(epoch_ious) / max(len(epoch_ious), 1),
            "baseline":      baseline(),
            "log_std":       policy.log_std.detach().cpu().tolist(),
            **val_metrics,
            "time_s":        elapsed,
        }
        rl_log.append(log_entry)

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"reward={log_entry['mean_reward']:.4f}  "
            f"val_iou={val_metrics['mean_iou']:.4f}  "
            f"rsa50={val_metrics['rsa_50']:.4f}  "
            f"baseline={baseline():.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["mean_iou"] > best_iou:
            best_iou = val_metrics["mean_iou"]
            policy.save_checkpoint(
                os.path.join(args.output_dir, "best.pt"),
                meta={"epoch": epoch, **val_metrics},
            )
            print(f"  → New best (IoU={best_iou:.4f}), saved best.pt")

        with open(os.path.join(args.output_dir, "rl_log.json"), "w") as f:
            json.dump(rl_log, f, indent=2)

    policy.save_checkpoint(
        os.path.join(args.output_dir, "final.pt"),
        meta={"epoch": args.epochs},
    )
    print(f"\nRL training complete. Best val IoU: {best_iou:.4f}")


@torch.no_grad()
def _validate_greedy(policy, val_ds, processor, device, batch_size):
    """Validate using greedy (mean) box predictions."""
    policy.eval()
    collate = make_collate(processor, device)
    loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    ious, rsa5, rsa25, areas = [], [], [], []

    for batch in loader:
        if batch is None:
            continue
        raw_mean = policy.forward_mean_raw(
            batch["pixel_values"], batch["input_ids"], batch["attention_mask"]
        )
        boxes = raw_to_box(raw_mean)
        for pb, tb in zip(boxes.cpu().tolist(), batch["target_boxes"].cpu().tolist()):
            iou_val = _box_iou_scalar(pb, tb)
            ious.append(iou_val)
            rsa5.append(1.0 if iou_val >= 0.5 else 0.0)
            rsa25.append(1.0 if iou_val >= 0.25 else 0.0)
            areas.append(max(0.0, (pb[2]-pb[0])*(pb[3]-pb[1])))

    n = len(ious) or 1
    return {
        "mean_iou":  sum(ious) / n,
        "rsa_50":    sum(rsa5) / n,
        "rsa_25":    sum(rsa25) / n,
        "mean_area": sum(areas) / n,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="RL fine-tuning of visual guidance policy.")
    p.add_argument("--data",            default="data/vsr_guidance_full.json")
    p.add_argument("--sft-checkpoint",  default="../checkpoints/guidance_sft/best.pt")
    p.add_argument("--reward-type",     default="grounding",
                   choices=["grounding", "combined", "answer_only"])
    p.add_argument("--model",           default="qwen",
                   help="VLM for answer reward (only for combined mode)")
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch-size",      type=int,   default=32)
    p.add_argument("--samples",         type=int,   default=4,
                   help="Number of box samples per example per step")
    p.add_argument("--max-train",       type=int,   default=None,
                   help="Cap training examples (use ~200 for combined mode to limit VLM calls)")
    p.add_argument("--lr",              type=float, default=1e-5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--output-dir",      default="../checkpoints/guidance_rl")
    args = p.parse_args()
    train_rl(args)


if __name__ == "__main__":
    main()
