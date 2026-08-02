# LTX model roles

This directory contains checkpoint-compatible LTX transformer architecture code only. Runtime assembly, scheduling, loading, offload, quantization policy, and denoising loops live in the shared `diffusion_model` infrastructure.

The architecture is adapted from [Lightricks/LTX-2](https://github.com/Lightricks/LTX-2) at revision `d6053703e00195bc668cbd1d5eda9dc0b2e7b74a`. LTX checkpoints and adapted source remain subject to the upstream LTX license terms. AlayaWorld's action/history model and compile adapter live with that synthesis product and reuse this architecture.
