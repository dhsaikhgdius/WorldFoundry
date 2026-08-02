def __getattr__(name):
    if name in {"HYVideoDiffusionTransformer", "HUNYUAN_VIDEO_CONFIG"}:
        from .models import HUNYUAN_VIDEO_CONFIG, HYVideoDiffusionTransformer

        value = {
            "HYVideoDiffusionTransformer": HYVideoDiffusionTransformer,
            "HUNYUAN_VIDEO_CONFIG": HUNYUAN_VIDEO_CONFIG,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(name)


def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG

    if args.model not in HUNYUAN_VIDEO_CONFIG:
        raise NotImplementedError(f"unsupported HunyuanVideo model variant: {args.model}")
    return HYVideoDiffusionTransformer(
        args,
        in_channels=in_channels,
        out_channels=out_channels,
        **HUNYUAN_VIDEO_CONFIG[args.model],
        **factor_kwargs,
    )
