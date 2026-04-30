"""
Modular VLM interface for Qwen2.5-VL and future backends.

Today's implementation: QwenVLLocalInterface (HuggingFace transformers)

Future stubs (not implemented today):
  - QwenVLvLLMInterface  : vLLM backend
  - APIInterface         : generic OpenAI-compatible or Anthropic API

Usage:
    from model_interface import QwenVLLocalInterface
    model = QwenVLLocalInterface()
    response = model.generate_response(pil_image, prompt_string)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VLMInterface(ABC):
    """Common interface every backend must implement."""

    @abstractmethod
    def generate_response(self, image: Image.Image, prompt: str) -> str:
        """
        Run a single inference pass.

        Args:
            image  : PIL RGB image
            prompt : fully-formatted text prompt (caption already inserted)
        Returns:
            Raw string output from the model (not yet parsed).
        """
        pass

    def generate_response_multi(self, images: list[Image.Image], prompt: str) -> str:
        """
        Run inference with multiple images in one prompt.

        Default implementation falls back to the first image only.
        Backends that support multi-image input should override this.

        Args:
            images : list of PIL RGB images (e.g. [full_image, crop])
            prompt : fully-formatted text prompt
        Returns:
            Raw string output from the model.
        """
        return self.generate_response(images[0], prompt)


# ---------------------------------------------------------------------------
# Local HuggingFace backend  (PRIMARY TODAY)
# ---------------------------------------------------------------------------

class QwenVLLocalInterface(VLMInterface):
    """
    Local Qwen2.5-VL-7B-Instruct inference via HuggingFace transformers.

    Requirements:
        pip install transformers accelerate
        pip install qwen-vl-utils          # optional, provides process_vision_info

    Model is auto-downloaded from HuggingFace Hub on first run (~15 GB).
    Needs ~14 GB VRAM in bfloat16. Falls back to CPU/float32 if no GPU.

    Generation is greedy (do_sample=False) for reproducibility.
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        torch_dtype=None,
        max_new_tokens: int = 512,
    ):
        import torch

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise ImportError(
                "Missing dependencies. Run: pip install transformers accelerate"
            )

        self.max_new_tokens = max_new_tokens

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"Loading {model_name} (dtype={torch_dtype}, device_map={device_map}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        print("Model ready.")

        # Check for optional qwen_vl_utils
        try:
            from qwen_vl_utils import process_vision_info as _pvi
            self._process_vision_info = _pvi
        except ImportError:
            self._process_vision_info = None

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        """Run one image+text inference pass, return raw decoded string."""
        import torch

        image = image.convert("RGB")  # ensure correct mode

        # Build Qwen chat-format message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template → text string
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize and prepare image tensors
        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Fallback: pass PIL image directly
            inputs = self.processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Strip the prompt tokens; decode only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    def generate_response_multi(self, images: list[Image.Image], prompt: str) -> str:
        """
        Run inference with multiple images in a single prompt.

        Builds a single user message where each image gets its own
        {"type": "image"} entry followed by the text prompt.  Qwen2.5-VL
        can attend to all images jointly.

        Args:
            images : list of PIL RGB images, e.g. [full_image, crop]
            prompt : text prompt (caption already substituted)
        Returns:
            Raw decoded string from the model.
        """
        import torch

        images = [img.convert("RGB") for img in images]

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        if self._process_vision_info is not None:
            image_inputs, video_inputs = self._process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=[text], images=images,
                padding=True, return_tensors="pt",
            )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        input_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


# ---------------------------------------------------------------------------
# vLLM backend stub (tomorrow)
# ---------------------------------------------------------------------------

class QwenVLvLLMInterface(VLMInterface):
    """
    Stub: vLLM-based inference for higher throughput.

    TODO tomorrow:
      - Use vllm.LLM with SamplingParams(temperature=0, max_tokens=512)
      - Encode image as base64 or use vLLM multimodal inputs
    """

    def __init__(self, model_name: str = QwenVLLocalInterface.DEFAULT_MODEL, **kwargs):
        raise NotImplementedError(
            "vLLM backend not yet implemented. Use QwenVLLocalInterface."
        )

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# API backend stub (tomorrow)
# ---------------------------------------------------------------------------

class APIInterface(VLMInterface):
    """
    Stub: OpenAI-compatible or Anthropic API wrapper.

    TODO tomorrow:
      - Encode image as base64 data URL
      - Call API with messages in OpenAI vision format
    """

    def __init__(self, api_url: str, api_key: str, model_name: str, **kwargs):
        raise NotImplementedError(
            "API interface not yet implemented. Use QwenVLLocalInterface."
        )

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(backend: str = "local", **kwargs) -> VLMInterface:
    """
    Instantiate the right backend by name.

    Args:
        backend : 'local' | 'vllm' | 'api'
        **kwargs: forwarded to the backend constructor
    """
    backends = {
        "local": QwenVLLocalInterface,
        "vllm": QwenVLvLLMInterface,
        "api": APIInterface,
    }
    if backend not in backends:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(backends)}")
    return backends[backend](**kwargs)
