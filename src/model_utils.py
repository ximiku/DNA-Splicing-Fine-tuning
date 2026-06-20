"""Model compatibility helpers."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import torch


def disable_remote_flash_attention(model: torch.nn.Module) -> None:
    """Disable DNABERT2's remote Triton flash attention for modern Triton compatibility."""
    module_names = {model.__class__.__module__}
    bert = getattr(model, "bert", None)
    if bert is not None:
        module_names.add(bert.__class__.__module__)
        encoder = getattr(bert, "encoder", None)
        layers = getattr(encoder, "layer", []) if encoder is not None else []
        if layers:
            module_names.add(layers[0].__class__.__module__)
            module_names.add(layers[0].attention.self.__class__.__module__)
    disabled = []
    for module_name in module_names:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "flash_attn_qkvpacked_func"):
            setattr(module, "flash_attn_qkvpacked_func", None)
            disabled.append(module_name)
    if disabled:
        logging.info("Disabled remote Triton flash attention in modules: %s", sorted(disabled))


def copy_remote_model_code(model: torch.nn.Module, output_dir: str | Path) -> None:
    """Copy HuggingFace remote-code modules into a saved model directory."""
    module = sys.modules.get(model.__class__.__module__)
    if module is None or not getattr(module, "__file__", None):
        return
    source_dir = Path(module.__file__).resolve().parent
    output_dir = Path(output_dir)
    for filename in ["bert_layers.py", "bert_padding.py", "flash_attn_triton.py", "configuration_bert.py"]:
        source = source_dir / filename
        if source.exists():
            shutil.copy2(source, output_dir / filename)
            logging.info("Copied remote model code file %s to %s", source.name, output_dir)
