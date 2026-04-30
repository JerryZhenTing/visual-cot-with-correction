"""
Lightweight visual guidance policy for VSR relation-region prediction.

Architecture:
  - Frozen CLIP ViT-B/32 image encoder
  - Frozen CLIP text encoder
  - Concatenate image + text embeddings (512 + 512 = 1024)
  - Small MLP → 4 raw values → valid normalized box

Box parameterization (cx/cy/w/h → xyxy):
  cx = sigmoid(raw_cx)
  cy = sigmoid(raw_cy)
  w  = sigmoid(raw_w)  * (1 - MIN_WH) + MIN_WH
  h  = sigmoid(raw_h)  * (1 - MIN_WH) + MIN_WH
  xmin = clamp(cx - w/2, 0, 1)
  ymin = clamp(cy - h/2, 0, 1)
  xmax = clamp(cx + w/2, 0, 1)
  ymax = clamp(cy + h/2, 0, 1)

Usage:
    model = GuidancePolicy.from_pretrained("openai/clip-vit-base-patch32")
    boxes = model(images, texts)   # (B, 4) normalized xyxy
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

MIN_WH = 0.02   # minimum predicted width/height to avoid degenerate boxes
CLIP_DIM = 512  # ViT-B/32 embedding dimension


class BoxHead(nn.Module):
    """MLP that maps concatenated embeddings to raw box parameters."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GuidancePolicy(nn.Module):
    """
    Visual guidance policy: (image, text) → predicted relation box.

    CLIP encoders are frozen; only BoxHead parameters are trained.
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        clip = CLIPModel.from_pretrained(clip_model_name)
        self.vision_model = clip.vision_model
        self.text_model   = clip.text_model
        self.visual_proj  = clip.visual_projection
        self.text_proj    = clip.text_projection
        self.processor    = CLIPProcessor.from_pretrained(clip_model_name)

        # Freeze CLIP
        for p in self.vision_model.parameters():
            p.requires_grad_(False)
        for p in self.text_model.parameters():
            p.requires_grad_(False)
        for p in self.visual_proj.parameters():
            p.requires_grad_(False)
        for p in self.text_proj.parameters():
            p.requires_grad_(False)

        self.box_head = BoxHead(input_dim=CLIP_DIM * 2, hidden_dim=512)

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return normalized image embeddings (B, CLIP_DIM)."""
        vis_out = self.vision_model(pixel_values=pixel_values)
        emb = self.visual_proj(vis_out.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_texts(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return normalized text embeddings (B, CLIP_DIM)."""
        txt_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.text_proj(txt_out.pooler_output)
        return emb / emb.norm(dim=-1, keepdim=True)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            pixel_values  : (B, 3, 224, 224) preprocessed by CLIPProcessor
            input_ids     : (B, L) tokenized captions
            attention_mask: (B, L)
        Returns:
            boxes: (B, 4) normalized xyxy in [0, 1]
        """
        img_emb  = self.encode_images(pixel_values)
        txt_emb  = self.encode_texts(input_ids, attention_mask)
        combined = torch.cat([img_emb, txt_emb], dim=-1)
        raw      = self.box_head(combined)
        return raw_to_box(raw)

    def predict(
        self,
        images: list[Image.Image],
        texts: list[str],
        device: Optional[torch.device] = None,
    ) -> list[list[float]]:
        """
        Convenience method for inference without manually batching tensors.

        Args:
            images: list of PIL Images
            texts:  list of caption strings (same length as images)
            device: optional device override
        Returns:
            list of [xmin, ymin, xmax, ymax] boxes (one per image)
        """
        if device is None:
            device = next(self.parameters()).device

        inputs = self.processor(
            text=texts, images=images,
            return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        self.eval()
        with torch.no_grad():
            boxes = self.forward(
                inputs["pixel_values"],
                inputs["input_ids"],
                inputs["attention_mask"],
            )
        return boxes.cpu().tolist()

    @classmethod
    def from_pretrained(
        cls,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ) -> "GuidancePolicy":
        return cls(clip_model_name=clip_model_name)

    def save_checkpoint(self, path: str, meta: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "box_head_state_dict": self.box_head.state_dict(),
            "clip_model_name": self.processor.name_or_path,
            "meta": meta or {},
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> "GuidancePolicy":
        ckpt = torch.load(path, map_location=device or "cpu")
        model = cls(clip_model_name=ckpt.get("clip_model_name", "openai/clip-vit-base-patch32"))
        model.box_head.load_state_dict(ckpt["box_head_state_dict"])
        if device is not None:
            model = model.to(device)
        return model


# ---------------------------------------------------------------------------
# Box conversion
# ---------------------------------------------------------------------------

def raw_to_box(raw: torch.Tensor) -> torch.Tensor:
    """
    Convert unconstrained raw values (B, 4) to valid normalized xyxy boxes (B, 4).

    raw[:, 0] → cx via sigmoid
    raw[:, 1] → cy via sigmoid
    raw[:, 2] → w  via sigmoid, rescaled to [MIN_WH, 1]
    raw[:, 3] → h  via sigmoid, rescaled to [MIN_WH, 1]
    """
    cx = torch.sigmoid(raw[:, 0])
    cy = torch.sigmoid(raw[:, 1])
    w  = torch.sigmoid(raw[:, 2]) * (1.0 - MIN_WH) + MIN_WH
    h  = torch.sigmoid(raw[:, 3]) * (1.0 - MIN_WH) + MIN_WH

    xmin = torch.clamp(cx - w / 2, 0.0, 1.0)
    ymin = torch.clamp(cy - h / 2, 0.0, 1.0)
    xmax = torch.clamp(cx + w / 2, 0.0, 1.0)
    ymax = torch.clamp(cy + h / 2, 0.0, 1.0)

    # Guarantee xmin < xmax, ymin < ymax after clamping
    xmin, xmax = torch.min(xmin, xmax), torch.max(xmin, xmax)
    ymin, ymax = torch.min(ymin, ymax), torch.max(ymin, ymax)

    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def collate_fn(batch: list[dict], processor, device: Optional[torch.device] = None):
    """
    Collate a list of dataset items into model-ready tensors.

    Returns dict with pixel_values, input_ids, attention_mask,
    target_boxes, answers, example_ids.
    """
    images  = [item["image"] for item in batch]
    texts   = [item["caption"] for item in batch]
    targets = [item["target_box"] for item in batch]

    inputs = processor(
        text=texts, images=images,
        return_tensors="pt", padding=True, truncation=True, max_length=77,
    )
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    if device is not None:
        target_tensor = target_tensor.to(device)

    return {
        **inputs,
        "target_boxes": target_tensor,
        "answers":      [item["answer"]     for item in batch],
        "example_ids":  [item["example_id"] for item in batch],
        "image_paths":  [item["image_path"] for item in batch],
    }
