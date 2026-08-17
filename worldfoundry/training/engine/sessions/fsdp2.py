"""FSDP2 specialization of the native training session lifecycle."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from worldfoundry.training.api.contracts import TrainingBatch
from worldfoundry.training.distributed.parallel import DistributedTrainingContext
from worldfoundry.training.recipes.post_training.recipe import PostTrainingRecipe
from worldfoundry.training.recipes.spec import TrainingRecipe
from worldfoundry.training.tuning.application import AdapterApplication
from worldfoundry.training.tuning.peft import PeftLoraApplication

from ..fsdp import FSDP2TrainEngine
from .single_device import SingleDeviceTrainingSession


class FSDP2TrainingSession(SingleDeviceTrainingSession):
    """Own a torchrun-launched FSDP2 run while preserving exact DCP state."""

    def __init__(
        self,
        *,
        recipe: TrainingRecipe | PostTrainingRecipe,
        engine: FSDP2TrainEngine,
        dataloader: Iterable[TrainingBatch],
        distributed_context: DistributedTrainingContext,
        output_dir: str | Path | None = None,
        peft_application: PeftLoraApplication | None = None,
        adapter_application: AdapterApplication | None = None,
        data_identity: Mapping[str, object] | None = None,
        lr_scheduler: object | None = None,
        ema: object | None = None,
        export_ema: bool = False,
    ) -> None:
        if not isinstance(engine, FSDP2TrainEngine):
            raise TypeError("engine must be FSDP2TrainEngine")
        super().__init__(
            recipe=recipe,
            engine=engine,
            dataloader=dataloader,
            output_dir=output_dir,
            peft_application=peft_application,
            adapter_application=adapter_application,
            data_identity=data_identity,
            distributed_context=distributed_context,
            lr_scheduler=lr_scheduler,
            ema=ema,
            export_ema=export_ema,
        )


__all__ = ["FSDP2TrainingSession"]
