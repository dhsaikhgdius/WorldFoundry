# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> mast3r -> model.py functionality."""

import ast
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape  # noqa


inf = float('inf')

# Modified by WorldFoundry: the helpers below replace the original
# ``net = eval(ckpt['args'].model)`` in ``load_model`` with a whitelist-based
# constructor dispatch so that a tampered checkpoint string cannot execute
# arbitrary code (plan/code_review/11_vendored_integration.md [VI-22]).
_CHECKPOINT_LITERAL_NAMES = {"inf": inf, "nan": float("nan")}


class _CheckpointLiteralNames(ast.NodeTransformer):
    """Rewrite bare ``inf``/``nan`` names used by MASt3R args strings into constants."""

    def visit_Name(self, node):
        if node.id in _CHECKPOINT_LITERAL_NAMES:
            return ast.copy_location(ast.Constant(_CHECKPOINT_LITERAL_NAMES[node.id]), node)
        return node


def _checkpoint_call_kwargs(call, class_name):
    """Evaluate the keyword arguments of a checkpoint constructor call as literals only."""
    if call.args:
        raise ValueError(
            f"checkpoint field 'args.model' passes positional arguments to {class_name}; "
            "only keyword arguments with literal values are allowed"
        )
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError(
                f"checkpoint field 'args.model' uses **kwargs expansion for {class_name}; refusing to evaluate"
            )
        value_node = _CheckpointLiteralNames().visit(keyword.value)
        try:
            kwargs[keyword.arg] = ast.literal_eval(value_node)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"checkpoint field 'args.model' has a non-literal value for keyword "
                f"{keyword.arg!r} of {class_name}"
            ) from exc
    return kwargs


def _instantiate_model_from_checkpoint_args(args):
    """Instantiate a whitelisted model class from the checkpoint ``args.model`` string."""
    allowed_classes = {
        "AsymmetricMASt3R": AsymmetricMASt3R,
        "AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo,
    }
    try:
        expression = ast.parse(args, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"checkpoint field 'args.model' is not a valid constructor call: {args!r}") from exc
    call = expression.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError(f"checkpoint field 'args.model' must be a plain constructor call, got: {args!r}")
    model_class = allowed_classes.get(call.func.id)
    if model_class is None:
        raise ValueError(
            f"checkpoint field 'args.model' requests class {call.func.id!r} which is not in the "
            f"allowed set {sorted(allowed_classes)}"
        )
    return model_class(**_checkpoint_call_kwargs(call, call.func.id))


def load_model(model_path, device, verbose=True):
    """Load model.

    Args:
        model_path: The model path.
        device: The device.
        verbose: The verbose.
    """
    if verbose:
        print('... loading model from', model_path)
    # Modified by WorldFoundry: was a bare ``torch.load(model_path, map_location='cpu')``,
    # which under torch>=2.6 defaults to weights_only=True and fails on real MASt3R
    # checkpoints (argparse.Namespace under 'args'). Route through the central safe
    # loader: weights_only=True is attempted first and the unsafe-pickle fallback is
    # explicit ([VI-22]).
    from worldfoundry.core.model_loading.file import load_torch_checkpoint

    ckpt = load_torch_checkpoint(model_path, map_location='cpu', allow_unsafe_pickle_fallback=True)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    # Modified by WorldFoundry: was ``net = eval(args)`` (arbitrary code execution
    # from a downloaded checkpoint); see _instantiate_model_from_checkpoint_args.
    net = _instantiate_model_from_checkpoint_args(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    """Asymmetric ma st r implementation."""
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, use_offsets=False, sh_degree=1, **kwargs):
        """Init.

        Args:
            desc_mode: The desc mode.
            two_confs: The two confs.
            desc_conf_mode: The desc conf mode.
            use_offsets: The use offsets.
            sh_degree: The sh degree.
        """
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.use_offsets = use_offsets
        self.sh_degree = sh_degree
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """From pretrained.

        Args:
            pretrained_model_name_or_path: The pretrained model name or path.
        """
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        """Set downstream head.

        Args:
            output_mode: The output mode.
            head_type: The head type.
            landscape_only: The landscape only.
            depth_mode: The depth mode.
            conf_mode: The conf mode.
            patch_size: The patch size.
            img_size: The img size.
        """
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), use_offsets=self.use_offsets, sh_degree=self.sh_degree)
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode), use_offsets=self.use_offsets, sh_degree=self.sh_degree)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
