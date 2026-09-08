"""Unit tests for Runner MPS model-loading behavior.

Copyright 2025-2026 Fujitsu Ltd.

Author: Yuhki Yano
"""

from unittest.mock import MagicMock

import torch

from onecomp.runner import Runner


def _mock_model():
    model = MagicMock(name="model")
    param = torch.nn.Parameter(torch.empty(1))
    model.parameters.side_effect = lambda: iter([param])
    model.to.return_value = model
    return model, param


def test_quantize_with_calibration_loads_mps_model_on_cpu_then_moves_to_mps():
    """MPS avoids device_map='mps' for large sharded checkpoints."""
    model, param = _mock_model()

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
    runner.prepare_calibration_dataset.assert_called_once_with(param.device, model=model)
    quantizer.setup.assert_called_once_with(model)
    model.assert_called_once_with(**inputs)
    quantizer.execute_post_processing.assert_called_once_with()


def test_quantize_with_calibration_non_mps_uses_default_load_model():
    model, param = _mock_model()

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
    runner.prepare_calibration_dataset.assert_called_once_with(param.device, model=model)
    quantizer.setup.assert_called_once_with(model)
    model.assert_called_once_with(**inputs)
    quantizer.execute_post_processing.assert_called_once_with()
