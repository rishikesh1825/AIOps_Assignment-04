"""
captioner.py - CPU-only image captioning with BLIP, plus safe image loading.

Kept free of Streamlit and Prometheus code so it can be tested on its own.
"""
from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_NAME = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
MAX_NEW_TOKENS = int(os.getenv("CAPTION_MAX_TOKENS", "30"))
MAX_PIXELS = 40_000_000   # refuse anything larger than ~40 MP (decompression-bomb guard)
MAX_SIDE = 1024           # downscale before the processor; BLIP works at 384px anyway

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


class InvalidImageError(Exception):
    """Raised when uploaded bytes are not a usable image. `reason` is a metric label."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class CaptionResult:
    text: str
    preprocess_s: float
    inference_s: float


def load_image(data: bytes) -> tuple[Image.Image, dict]:
    """Decode and validate image bytes. Returns an RGB image and its original size info."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width * height > MAX_PIXELS:
                raise InvalidImageError(
                    "too_large", f"Image is {width}x{height}; the limit is {MAX_PIXELS / 1e6:.0f} MP."
                )
            probe.verify()  # catches truncated / corrupt files cheaply
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)  # respect phone-camera rotation
        image = image.convert("RGB")
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as exc:
        raise InvalidImageError("too_large", "Image has too many pixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError(
            "corrupt_image", f"Not a readable image ({exc.__class__.__name__})."
        ) from exc

    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image, {"width": width, "height": height, "megapixels": width * height / 1e6}


def make_thumbnail(image: Image.Image, size: int = 256) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    return thumb


class Captioner:
    """Loads BLIP once and generates captions on the CPU."""

    def __init__(self, model_name: str = MODEL_NAME, max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        threads = int(os.getenv("TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        start = time.perf_counter()
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.load_seconds = time.perf_counter() - start

        self._torch = torch
        self._lock = threading.Lock()  # one inference at a time; avoids CPU thrashing
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch_threads = torch.get_num_threads()
        self.torch_version = torch.__version__

        params = list(self.model.parameters())
        buffers = list(self.model.buffers())
        self.num_parameters = sum(p.numel() for p in params)
        self.memory_bytes = sum(t.numel() * t.element_size() for t in params + buffers)

    def caption(self, image: Image.Image) -> CaptionResult:
        with self._lock:
            t0 = time.perf_counter()
            inputs = self.processor(images=image, return_tensors="pt")
            t1 = time.perf_counter()
            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            t2 = time.perf_counter()
        text = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return CaptionResult(text=text, preprocess_s=t1 - t0, inference_s=t2 - t1)
