from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from worldfoundry.core.model_loading import load_torch_state_dict


class LingBotTorchCompatTest(unittest.TestCase):
    def test_load_torch_state_dict_prefers_weights_only(self) -> None:
        calls = []
        sentinel = {"weights": torch.tensor([1.0])}

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return sentinel

        with patch.object(torch, "load", side_effect=fake_load):
            loaded = load_torch_state_dict("/tmp/lingbot.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu", "weights_only": True},
                )
            ],
        )

    def test_load_torch_state_dict_falls_back_without_weights_only(self) -> None:
        calls = []
        sentinel = {"weights": torch.tensor([2.0])}

        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            if "weights_only" in kwargs:
                raise TypeError("load() got an unexpected keyword argument 'weights_only'")
            return sentinel

        with patch.object(torch, "load", side_effect=fake_load):
            loaded = load_torch_state_dict("/tmp/lingbot.pt", map_location="cpu")

        self.assertIs(loaded, sentinel)
        self.assertEqual(
            calls,
            [
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu", "weights_only": True},
                ),
                (
                    "/tmp/lingbot.pt",
                    {"map_location": "cpu"},
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
