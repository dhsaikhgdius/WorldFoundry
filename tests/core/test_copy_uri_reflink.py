from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from worldfoundry.core.io import storage as storage_mod


def test_copy_uri_local_delegates_to_materialize_file(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    target = tmp_path / "nested" / "dst.bin"
    source.write_bytes(b"payload-bytes")
    seen: dict[str, object] = {}

    def _fake_materialize(src, dst, *, writable=True):
        seen["src"] = str(src)
        seen["dst"] = str(dst)
        seen["writable"] = writable
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(Path(src).read_bytes())
        return "copy"

    with patch.object(storage_mod, "materialize_file", create=True):
        # Patch where copy_uri imports from (lazy import inside the function).
        with patch("worldfoundry.core.io.file_utils.materialize_file", side_effect=_fake_materialize) as mocked:
            result = storage_mod.copy_uri(str(source), str(target))
    assert result == str(target)
    assert target.read_bytes() == b"payload-bytes"
    assert seen["writable"] is True
    mocked.assert_called_once()
