# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
"""Module for base_models -> three_dimensions -> general_3d -> dust3r -> dust3r -> model.py functionality."""

import ast
from copy import deepcopy
import torch
import os
from packaging import version
import huggingface_hub

from .utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from .heads import head_factory
from dust3r.patch_embed import get_patch_embed, ManyAR_PatchEmbed

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

# Modified by WorldFoundry: the helpers below replace the original
# ``net = eval(ckpt['args'].model)`` in ``load_model`` with a whitelist-based
# constructor dispatch so that a tampered checkpoint string cannot execute
# arbitrary code (plan/code_review/11_vendored_integration.md [VI-22]).
_CHECKPOINT_LITERAL_NAMES = {"inf": inf, "nan": float("nan")}


class _CheckpointLiteralNames(ast.NodeTransformer):
    """Rewrite bare ``inf``/``nan`` names used by DUSt3R args strings into constants."""

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
    allowed_classes = {"AsymmetricCroCo3DStereo": AsymmetricCroCo3DStereo}
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
    # Modified by WorldFoundry: was a bare ``torch.load(..., weights_only=False)``.
    # Route through the central safe loader, which tries weights_only=True first;
    # DUSt3R checkpoints store an argparse.Namespace under 'args', so the explicit
    # unsafe-pickle fallback is still required for real checkpoints ([VI-22]).
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


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/junyi/monst3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).   
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 **croco_kwargs):
        """Init.

        Args:
            output_mode: The output mode.
            head_type: The head type.
            depth_mode: The depth mode.
            conf_mode: The conf mode.
            freeze: The freeze.
            landscape_only: The landscape only.
            patch_embed_cls: The patch embed cls.
        """
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        """From pretrained.

        Args:
            pretrained_model_name_or_path: The pretrained model name or path.
        """
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        """Helper function to set patch embed.

        Args:
            img_size: The img size.
            patch_size: The patch size.
            enc_embed_dim: The enc embed dim.
        """
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        """Load state dict.

        Args:
            ckpt: The ckpt.
        """
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        """Set freeze.

        Args:
            freeze: The freeze.
        """
        self.freeze = freeze
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
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
        if type(img_size) is int:
            img_size = (img_size, img_size)
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def _encode_image(self, image, true_shape):
        """Helper function to encode image.

        Args:
            image: The image.
            true_shape: The true shape.
        """
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x (B, 576, 1024) pos (B, 576, 2); patch_size=16
        B,N,C = x.size()
        posvis = pos
        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # TODO: where to add mask for the patches
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, posvis)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        """Helper function to encode image pairs.

        Args:
            img1: The img1.
            img2: The img2.
            true_shape1: The true shape1.
            true_shape2: The true shape2.
        """
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1)
            out2, pos2, _ = self._encode_image(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        """Helper function to encode symmetrized.

        Args:
            view1: The view1.
            view2: The view2.
        """
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]

        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2):
        """Helper function to decoder.

        Args:
            f1: The f1.
            pos1: The pos1.
            f2: The f2.
            pos2: The pos2.
        """
        final_output = [(f1, f2)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        """Helper function to downstream head.

        Args:
            head_num: The head num.
            decout: The decout.
            img_shape: The img shape.
        """
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        """Forward.

        Args:
            view1: The view1.
            view2: The view2.
        """
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2
