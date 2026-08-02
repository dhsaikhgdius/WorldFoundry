"""Framework-neutral prompt preprocessing hooks used by inference components."""

from __future__ import annotations

from collections.abc import Sequence

import torch


class PromptProcessor:
    """Apply optional prompt refiners/extenders without owning model loading."""

    def __init__(self) -> None:
        self.refiners = []
        self.extenders = []

    def load_prompt_refiners(self, model_source, refiner_classes: Sequence[type] = ()) -> None:
        for refiner_class in refiner_classes:
            self.refiners.append(refiner_class.from_model_manager(model_source))

    def load_prompt_extenders(self, model_source, extender_classes: Sequence[type] = ()) -> None:
        for extender_class in extender_classes:
            self.extenders.append(extender_class.from_model_manager(model_source))

    @torch.no_grad()
    def process_prompt(self, prompt, positive: bool = True):
        if isinstance(prompt, list):
            return [self.process_prompt(item, positive=positive) for item in prompt]
        for refiner in self.refiners:
            prompt = refiner(prompt, positive=positive)
        return prompt

    @torch.no_grad()
    def extend_prompt(self, prompt: str, positive: bool = True):
        del positive
        extended = {"prompt": prompt}
        for extender in self.extenders:
            extended = extender(extended)
        return extended


__all__ = ["PromptProcessor"]
