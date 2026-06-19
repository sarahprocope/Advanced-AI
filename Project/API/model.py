from __future__ import annotations

import logging
import os
import torch
from PIL import Image

from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

logger = logging.getLogger(__name__)

# Configurable checkpoint path via environment variable
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints/best_step1000")

_model: VisionLanguageModel | None = None
_tokenizer = None
_image_processor = None
_device = "cpu"


def load_model() -> None:
    global _model, _tokenizer, _image_processor, _device
    if _model is not None:
        return

    if torch.cuda.is_available():
        _device = "cuda"
        dtype = torch.bfloat16
    else:
        _device = "cpu"
        dtype = torch.float32

    logger.info("Loading custom VLM checkpoint from %s on %s (dtype=%s)", CHECKPOINT_DIR, _device, dtype)
    
    if not os.path.exists(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint directory {CHECKPOINT_DIR} not found. Ensure checkpoints volume is mounted.")

    _model = VisionLanguageModel.from_pretrained(CHECKPOINT_DIR).to(_device)
    _model.eval()

    cfg = _model.cfg
    _tokenizer = get_tokenizer(cfg.lm.tokenizer, cfg.image_token)
    _image_processor = get_image_processor(cfg.vit.img_size)
    logger.info("Custom VLM loaded successfully")


def generate(
    messages: list[dict],
    image: Image.Image | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    if _model is None or _tokenizer is None or _image_processor is None:
        raise RuntimeError("Model not loaded; call load_model() first")

    cfg = _model.cfg
    
    # If no image is provided, use a dummy black image
    if image is None:
        logger.info("No image provided. Creating a dummy black image.")
        image = Image.new("RGB", (512, 512), (0, 0, 0))

    # Preprocess the image
    pixel_values = _image_processor(image).unsqueeze(0).to(_device)

    # Build the prompt with image placeholders prepended to the first user message
    image_string = get_image_string(
        cfg.projector.image_token_length, cfg.image_token
    )
    
    formatted: list[dict] = []
    prepended = False
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user" and not prepended:
            content = image_string + content
            prepended = True
        formatted.append({"role": role, "content": content})

    # Apply the chat template and tokenize
    encoded = _tokenizer.apply_chat_template(
        formatted, tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor(encoded).unsqueeze(0).to(_device)
    attention_mask = torch.ones_like(input_ids)

    # Run custom model generation
    do_sample = temperature > 0.0
    with torch.no_grad():
        gen = _model.generate(
            input_ids,
            pixel_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            greedy=not do_sample,
            temperature=temperature if do_sample else 1.0,
        )

    # Decode response
    text = _tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()
