from worldfoundry.synthesis.action_generation.diffusion_policy.modeling.mask import (
    LowdimMaskGenerator,
)
from worldfoundry.synthesis.action_generation.diffusion_policy.modeling.policy import (
    DiffusionUnetLowdimPolicy,
)


def test_lowdim_policy_is_available_directly_in_tree() -> None:
    assert DiffusionUnetLowdimPolicy.__module__.startswith("worldfoundry.")
    assert LowdimMaskGenerator.__module__.startswith("worldfoundry.")
