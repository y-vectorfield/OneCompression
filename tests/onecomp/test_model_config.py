"""Unit tests for ModelConfig MPS model-loading behavior.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from onecomp.model_config import ModelConfig


def _mock_loaded_model():
    model = MagicMock(name="model")
    model.parameters.side_effect = lambda: iter([torch.empty(1, dtype=torch.float16)])
    model.to.return_value = model
    return model


def test_load_model_loads_mps_model_on_cpu_then_moves_to_mps(monkeypatch):
    """MPS loads sharded checkpoints on CPU before moving the model to MPS."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)

    load_config = MagicMock(return_value=config)
    load_model = MagicMock(return_value=model)

    monkeypatch.setattr("onecomp.model_config.AutoConfig.from_pretrained", load_config)
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )

    model_config = ModelConfig(model_id="test/model", device="mps")
    loaded = model_config.load_model()

    assert loaded is model
    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map="cpu",
    )
    model.to.assert_called_once_with(torch.device("mps"))
    model.eval.assert_called_once_with()


def test_load_model_keeps_non_mps_device_placement(monkeypatch):
    """Non-MPS devices retain their requested device_map."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)

    monkeypatch.setattr(
        "onecomp.model_config.AutoConfig.from_pretrained",
        MagicMock(return_value=config),
    )
    load_model = MagicMock(return_value=model)
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )

    model_config = ModelConfig(model_id="test/model", device="cpu")
    model_config.load_model()

    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map="cpu",
    )
    model.to.assert_not_called()
    model.eval.assert_called_once_with()


def test_load_model_auto_uses_mps_workaround_when_mps_is_default(monkeypatch):
    """device='auto' uses the MPS workaround when MPS is selected."""
    model = _mock_loaded_model()
    config = SimpleNamespace(quantization_config=None)
    load_model = MagicMock(return_value=model)

    monkeypatch.setattr(
        "onecomp.model_config.AutoConfig.from_pretrained",
        MagicMock(return_value=config),
    )
    monkeypatch.setattr(
        "onecomp.model_config.AutoModelForCausalLM.from_pretrained",
        load_model,
    )
    monkeypatch.setattr(
        "onecomp.model_config.get_default_device",
        MagicMock(return_value=torch.device("mps")),
    )

    ModelConfig(model_id="test/model", device="auto").load_model()

    load_model.assert_called_once_with(
        "test/model",
        dtype=torch.float16,
        device_map="cpu",
    )
    model.to.assert_called_once_with(torch.device("mps"))
    model.eval.assert_called_once_with()
