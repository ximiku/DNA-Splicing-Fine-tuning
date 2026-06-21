"""Model compatibility helpers."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


def disable_remote_flash_attention(model: torch.nn.Module) -> None:
    """Disable DNABERT2's remote Triton flash attention for modern Triton compatibility."""
    if hasattr(model, "get_base_model"):
        try:
            model = model.get_base_model()
        except Exception:
            logging.debug("Could not unwrap PEFT base model before disabling flash attention")
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
    if hasattr(model, "get_base_model"):
        try:
            model = model.get_base_model()
        except Exception:
            logging.debug("Could not unwrap PEFT base model for remote-code copy")
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


def load_saved_sequence_classifier(
    model_dir: str | Path,
    *,
    trust_remote_code: bool = True,
    num_labels: int = 3,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
) -> torch.nn.Module:
    """Load a full fine-tuned model or a PEFT adapter directory for inference."""
    model_path = Path(model_dir)
    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "num_labels": num_labels,
    }
    if id2label is not None:
        load_kwargs["id2label"] = id2label
    if label2id is not None:
        load_kwargs["label2id"] = label2id

    if (model_path / "adapter_config.json").exists():
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:
            raise ImportError("Loading a PEFT adapter requires the peft package.") from exc
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            **load_kwargs,
        )
        return PeftModel.from_pretrained(base_model, str(model_path))

    return AutoModelForSequenceClassification.from_pretrained(str(model_path), **load_kwargs)
