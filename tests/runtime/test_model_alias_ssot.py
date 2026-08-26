"""I-12: model-id aliases have a single source of truth in runtime.conda.

The bash installer ``scripts/setup/model_env_install.sh`` must not carry its
own alias table; it queries ``python -m worldfoundry.runtime.conda
--canonical-model-id`` instead.
"""

from __future__ import annotations

import subprocess
import sys

from worldfoundry.core.io.paths import project_root
from worldfoundry.runtime.conda import _MODEL_ID_ALIASES, canonical_model_id

MODEL_ENV_INSTALL = project_root() / "scripts" / "setup" / "model_env_install.sh"


def test_python_aliases_cover_all_legacy_ids():
    assert canonical_model_id("lyra1") == "lyra-1"
    for alias in ("cosmos3-nano", "cosmos3-super", "cosmos-3", "cosmos-3-nano", "cosmos-3-super"):
        assert canonical_model_id(alias) == "cosmos3"


def test_unknown_and_padded_ids_normalize():
    assert canonical_model_id("wan2.1-t2v-1.3b") == "wan2.1-t2v-1.3b"
    assert canonical_model_id("  lyra1  ") == "lyra-1"
    assert canonical_model_id("") == ""


def test_module_cli_prints_one_canonical_id_per_input():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "worldfoundry.runtime.conda",
            "--canonical-model-id",
            "lyra1",
            "cosmos-3-nano",
            "evalcrafter",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root(),
    )
    assert completed.stdout.split() == ["lyra-1", "cosmos3", "evalcrafter"]


def test_bash_installer_queries_python_instead_of_owning_aliases():
    text = MODEL_ENV_INSTALL.read_text(encoding="utf-8")
    assert "canonical_model_id() {" not in text, "bash must not define its own alias table"
    assert "-m worldfoundry.runtime.conda --canonical-model-id" in text
    for canonical in set(_MODEL_ID_ALIASES.values()):
        assert f'printf \'%s\\n\' "{canonical}"' not in text, "leftover hardcoded alias output in bash"
