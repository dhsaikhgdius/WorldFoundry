# Cosmos 2.5 networks

Inference-only PyTorch implementation of NVIDIA Cosmos Predict 2.5's
`MinimalV1LVGDiT`. It uses WorldFoundry core attention/timestep primitives and
does not depend on Diffusers or the upstream training runtime. NVIDIA EMA
checkpoint conversion and the `Denoiser` contract adapter live in the shared
role `models/denoisers/cosmos2p5.py`, not in this network package.

`Cosmos25Transfer3DModel` extends the same checkpoint-shaped blocks with the
released Transfer 2.5 VACE layout: one 8-modality control embedder and four
control blocks mapped to base layers 0, 7, 14, and 21. The released checkpoint
also disables temporal FPS RoPE modulation. It is a denoiser variant in the
standard native recipe, not a separate ControlNet runtime. Reason1 conditioning,
Wan VAE encoding/decoding, flow-UniPC sampling, loading, and offload policy
remain shared framework components.

Upstream reference: `nvidia-cosmos/cosmos-predict2.5` at
`a2c298b0a3df3778b973fe65e9e58877b292d8a7`.
