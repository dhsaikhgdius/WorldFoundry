# Cosmos3 omni transformer

Inference-only native PyTorch implementation of the Cosmos3 joint text, video,
sound, and action transformer. The parameter layout matches the official
`nvidia/Cosmos3-Nano` and `nvidia/Cosmos3-Super` checkpoints directly.

The implementation is derived from NVIDIA's Cosmos3 Diffusers contribution at
repository revision `aefdf852c16237039785d84ab3327167cb0ec07f`. WorldFoundry
owns loading, attention dispatch, scheduling, modality packing, and execution
outside this network role; Diffusers is not a runtime backend.
