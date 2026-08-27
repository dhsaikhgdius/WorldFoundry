"""Gradio import guards and runtime patches for WorldFoundry Studio.

Importing this module imports Gradio (with SOCKS-proxy masking) but does not
mutate it. Studio entrypoints opt in to the runtime guards by calling
:func:`install_gradio_patches` before building the Gradio UI.
"""

from __future__ import annotations

import inspect
import os
import time
import warnings
from typing import Any

import httpx

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def mask_socks_proxy_env_for_gradio() -> dict[str, str]:
    """Avoid Gradio's local httpx probes failing when socksio is absent."""

    masked: dict[str, str] = {}
    for name in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        value = os.environ.get(name)
        if value and value.strip().lower().startswith(
            ("socks://", "socks4://", "socks4h://", "socks5://", "socks5h://")
        ):
            masked[name] = value
            os.environ.pop(name, None)
    return masked


_GRADIO_IMPORT_PROXY_ENV = mask_socks_proxy_env_for_gradio()
try:
    import gradio as gr
finally:
    os.environ.update(_GRADIO_IMPORT_PROXY_ENV)


def _install_template_response_guard() -> None:
    """Route Gradio 4.14's older TemplateResponse call through Starlette >= 1.0."""

    try:
        import gradio.routes as gr_routes
    except Exception:
        return

    templates = getattr(gr_routes, "templates", None)
    if templates is None or getattr(templates, "_worldfoundry_template_response_guard", False):
        return

    template_response = getattr(templates, "TemplateResponse", None)
    if template_response is None:
        return

    try:
        params = list(inspect.signature(template_response).parameters)
    except (TypeError, ValueError):
        return

    if not params or params[0] != "request":
        return

    original = template_response

    def guarded_template_response(*args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.get("context")
            request = context.get("request") if isinstance(context, dict) else None
            if request is None:
                raise TypeError(
                    "Gradio older TemplateResponse call requires context['request'] "
                    "when running with Starlette >= 1.0."
                )
            return original(request, name, context, *args[2:], **kwargs)
        return original(*args, **kwargs)

    templates.TemplateResponse = guarded_template_response
    templates._worldfoundry_template_response_guard = True


def _install_proxy_safe_url_check() -> None:
    """Make Gradio's localhost readiness probe ignore shell proxy settings."""

    try:
        import gradio.networking as gr_networking
    except Exception:
        return

    if getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False):
        return

    def proxy_safe_url_ok(url: str) -> bool:
        try:
            with httpx.Client(timeout=3, verify=False, trust_env=False) as client:
                for _ in range(5):
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        response = client.head(url)
                    if response.status_code in (200, 401, 302):
                        return True
                    time.sleep(0.5)
        except Exception:
            return False
        return False

    gr_networking.url_ok = proxy_safe_url_ok
    gr_networking._worldfoundry_proxy_safe_url_ok = True


def _install_api_info_guard() -> None:
    """Avoid Gradio API schema crashes on components that emit bool schemas."""

    try:
        import gradio.blocks as gr_blocks
    except Exception:
        return

    blocks_cls = getattr(gr_blocks, "Blocks", None)
    if blocks_cls is None or getattr(blocks_cls, "_worldfoundry_api_info_guard", False):
        return

    original = getattr(blocks_cls, "get_api_info", None)
    if original is None:
        return

    def guarded_get_api_info(self, all_endpoints: bool = False):
        try:
            return original(self, all_endpoints=all_endpoints)
        except TypeError as exc:
            if "all_endpoints" in str(exc) and "unexpected keyword" in str(exc):
                try:
                    return original(self)
                except TypeError as nested_exc:
                    exc = nested_exc
            error_text = str(exc)
            if "bool" not in error_text or "iterable" not in error_text:
                raise
            return {"named_endpoints": {}, "unnamed_endpoints": {}}

    blocks_cls.get_api_info = guarded_get_api_info
    blocks_cls._worldfoundry_api_info_guard = True


_EXPECTED_GRADIO_MAJOR_VERSIONS = (4, 5)


def _warn_on_unexpected_gradio_version() -> None:
    version = str(getattr(gr, "__version__", ""))
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError:
        major = None
    if major not in _EXPECTED_GRADIO_MAJOR_VERSIONS:
        warnings.warn(
            "WorldFoundry Studio's Gradio runtime guards target Gradio major versions "
            f"{_EXPECTED_GRADIO_MAJOR_VERSIONS} but found {version!r}; the patched hooks "
            "(templates.TemplateResponse, networking.url_ok, Blocks.get_api_info) may "
            "have drifted upstream.",
            RuntimeWarning,
            stacklevel=3,
        )


def install_gradio_patches() -> None:
    """Install WorldFoundry's Gradio runtime guards process-wide.

    Importing this module has no side effects on Gradio; Studio entrypoints
    must call this explicitly before building or launching the Gradio UI.
    Each guard marks its patch target with a ``_worldfoundry_*`` sentinel, so
    repeated calls are idempotent.
    """

    _warn_on_unexpected_gradio_version()
    _install_template_response_guard()
    _install_proxy_safe_url_check()
    _install_api_info_guard()


def gradio_progress(*, track_tqdm: bool = False) -> Any:
    progress_cls = getattr(gr, "Progress", None)
    if callable(progress_cls):
        return progress_cls(track_tqdm=track_tqdm)
    return None
