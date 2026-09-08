"""Unit tests for Runner MPS model-loading behavior.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano
"""

from unittest.mock import MagicMock

import torch

from onecomp.runner import Runner


def _mock_model():
    model = MagicMock(name="model")
    current_device = torch.device("cpu")

    def mock_to(device):
        nonlocal current_device
        current_device = torch.device(device)
        return model

    def mock_parameters():
        param = MagicMock(name="parameter")
        param.device = current_device
        return iter([param])

    model.to.side_effect = mock_to
    model.parameters.side_effect = mock_parameters
    return model


def test_quantize_with_calibration_loads_mps_model_on_cpu_then_moves_to_mps():
    """MPS avoids device_map='mps' for large sharded checkpoints."""
    model = _mock_model()

    model_config = MagicMock(name="model_config")
    model_config.get_device.return_value = torch.device("mps")
    model_config.load_model.return_value = model

    quantizer = MagicMock(name="quantizer")
    quantizer.name = "GPTQ_4bit"
    quantizer.module_to_name = {}

    inputs = {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        report_progress=False,
    )
    runner.prepare_calibration_dataset = MagicMock(return_value=inputs)

    runner.quantize_with_calibration()

    model_config.load_model.assert_called_once_with(device_map="cpu")
    model.to.assert_called_once_with("mps")
    runner.prepare_calibration_dataset.assert_called_once_with(
        torch.device("mps"),
        model=model,
    )
    quantizer.execute_post_processing.assert_called_once_with()


def test_quantize_with_calibration_non_mps_uses_default_load_model():
    """Non-MPS devices keep the default model-loading path."""
    model = _mock_model()

    model_config = MagicMock(name="model_config")
    model_config.get_device.return_value = torch.device("cpu")
    model_config.load_model.return_value = model

    quantizer = MagicMock(name="quantizer")
    quantizer.name = "GPTQ_4bit"
    quantizer.module_to_name = {}

    inputs = {"input_ids": torch.ones((1, 1), dtype=torch.long)}

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        report_progress=False,
    )
    runner.prepare_calibration_dataset = MagicMock(return_value=inputs)

    runner.quantize_with_calibration()

    model_config.load_model.assert_called_once_with()
    model.to.assert_not_called()
    runner.prepare_calibration_dataset.assert_called_once_with(
        torch.device("cpu"),
        model=model,
    )
    quantizer.setup.assert_called_once_with(model)
    model.assert_called_once_with(**inputs)
    quantizer.execute_post_processing.assert_called_once_with()
