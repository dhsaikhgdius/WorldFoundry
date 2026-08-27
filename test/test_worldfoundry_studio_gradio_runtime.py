from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# This test module imports worldfoundry code that requires the optional
# "gradio" dependency at import time; skip when it is unavailable.
pytest.importorskip("gradio")

import unittest

import gradio.blocks as gr_blocks
import gradio.networking as gr_networking
import gradio.routes as gr_routes

from worldfoundry.studio import gradio_runtime

_REPO_ROOT = Path(__file__).resolve().parents[1]

_IMPORT_ONLY_PROBE = """
import worldfoundry.studio.gradio_runtime  # noqa: F401

import gradio.blocks as gr_blocks
import gradio.networking as gr_networking
import gradio.routes as gr_routes

assert not getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False), (
    "importing gradio_runtime must not patch gradio.networking.url_ok"
)
assert not getattr(gr_blocks.Blocks, "_worldfoundry_api_info_guard", False), (
    "importing gradio_runtime must not patch Blocks.get_api_info"
)
templates = getattr(gr_routes, "templates", None)
assert templates is None or not getattr(
    templates, "_worldfoundry_template_response_guard", False
), "importing gradio_runtime must not patch templates.TemplateResponse"
print("import-side-effect-free")
"""


class WorldFoundryStudioGradioRuntimeTest(unittest.TestCase):
    def test_import_alone_does_not_install_patches(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH", "")])
        )
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_ONLY_PROBE],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("import-side-effect-free", result.stdout)

    def test_install_gradio_patches_sets_sentinels(self) -> None:
        gradio_runtime.install_gradio_patches()

        self.assertTrue(getattr(gr_networking, "_worldfoundry_proxy_safe_url_ok", False))
        self.assertTrue(getattr(gr_blocks.Blocks, "_worldfoundry_api_info_guard", False))
        # The template guard only applies when TemplateResponse exposes the
        # Starlette >= 1.0 (request-first) signature; when it applies its
        # sentinel must be set as well.
        templates = getattr(gr_routes, "templates", None)
        if templates is not None:
            import inspect

            params: list[str] = []
            template_response = getattr(templates, "TemplateResponse", None)
            if template_response is not None:
                try:
                    params = list(inspect.signature(template_response).parameters)
                except (TypeError, ValueError):
                    params = []
            if params and params[0] == "request":
                self.assertTrue(
                    getattr(templates, "_worldfoundry_template_response_guard", False)
                )

    def test_install_gradio_patches_is_idempotent(self) -> None:
        gradio_runtime.install_gradio_patches()
        url_ok_first = gr_networking.url_ok
        api_info_first = gr_blocks.Blocks.get_api_info
        templates = getattr(gr_routes, "templates", None)
        template_response_first = getattr(templates, "TemplateResponse", None)

        gradio_runtime.install_gradio_patches()

        self.assertIs(gr_networking.url_ok, url_ok_first)
        self.assertIs(gr_blocks.Blocks.get_api_info, api_info_first)
        # When the template guard applied, TemplateResponse is a plain function
        # stored on the templates instance and must not be re-wrapped. (When it
        # did not apply, the attribute stays a bound method whose identity
        # changes on every access, so identity cannot be asserted.)
        if templates is not None and getattr(
            templates, "_worldfoundry_template_response_guard", False
        ):
            self.assertIs(templates.TemplateResponse, template_response_first)


if __name__ == "__main__":
    unittest.main()
