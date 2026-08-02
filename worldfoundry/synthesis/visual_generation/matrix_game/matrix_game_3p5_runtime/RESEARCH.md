# Matrix-Game 3.5 memory-research extension contract

The released first- and third-person checkpoints remain ordinary independent
WorldFoundry models. Research mechanisms should be injected behind typed
dataset, retrieval, module, or denoiser boundaries; they should not add a
mutable `memory_method` switch to either public model ID.

## Extension seams

1. **Memory source and retrieval inputs**

   Subclass `DA3MosaicVideoDataset` or
   `SubjectRefMemoryDA3MosaicVideoDataset`, then pass the class to
   `build_mosaic_inference_dataset(..., dataset_cls=...)`. Keep the returned
   sample schema compatible with the Mosaic runner so retrieval experiments do
   not fork checkpoint loading or video I/O.

2. **Geometry and retrieval policy**

   Extend or compose `FrustumHandler` for candidate selection, coverage,
   reprojection, or fusion experiments. Preserve the explicit camera contract:
   pixel intrinsics, 4x4 extrinsics, and declared c2w/w2c convention. New state
   should be owned by the experiment object rather than module-level globals.

3. **Rollout orchestration**

   Subclass `WanMosaicPipelineModule` and inject it through
   `build_mosaic_pipeline_module(args, module_cls=...)`. This is the boundary
   for alternate read/write timing, history retention, or per-section memory
   policies.

4. **Denoiser-side memory representation**

   Add Matrix-specific tensor behavior to the canonical
   `models/networks/matrix_game_3p5` denoiser component and expose it through a
   new declarative recipe or typed extension. Reuse the shared runner,
   scheduler, loader, VAE, and text encoder; do not create a model-owned
   pipeline or fork the native infrastructure.

5. **Subject-reference memory**

   Keep subject memory as an explicit prefix-token contract. A new reference
   encoder or selection policy should preserve bounded reference counts,
   deterministic ordering, and the PRoPE/time-index metadata consumed by the
   canonical Matrix denoiser component.

## Minimal dataset experiment

```python
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.data import (
    DA3MosaicVideoDataset,
)
from worldfoundry.synthesis.visual_generation.matrix_game.matrix_game_3p5_runtime.mosaic.datasets import (
    build_mosaic_inference_dataset,
)


class RetrievalExperiment(DA3MosaicVideoDataset):
    """Example: add deterministic retrieval metadata without changing inference I/O."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["retrieval_experiment"] = {"name": "my-policy", "version": 1}
        return sample


dataset = build_mosaic_inference_dataset(args, dataset_cls=RetrievalExperiment)
```

Production experiments should expose the injected class through a fixed model
recipe or benchmark fixture rather than editing the released IDs in place.

## Promotion checklist

- Give the experiment a unique immutable model/variant ID and a revision-pinned
  checkpoint contract.
- Add its config under
  `worldfoundry/data/models/runtime/configs/matrix_game_3p5`, plus catalog,
  pipeline binding, runtime profile, and documentation entries.
- Validate camera shapes/conventions and exact checkpoint key/shape coverage.
- Add deterministic CPU contracts for retrieval and memory layout.
- Run a bounded CUDA smoke, then the declared default inference recipe.
- Record artifact hashes and distinguish route validation, benchmark scores,
  and official-sample parity.
- Document checkpoint and dependency licenses; DA3 weights are CC-BY-NC-4.0.

Training APIs are deliberately outside this infer-only surface.
