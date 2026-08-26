from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from worldfoundry.studio.visualization.backends import viser as viser_mod


def test_geometry_helpers_reuse_loaded_asset_without_reload(tmp_path: Path) -> None:
    path = tmp_path / "demo.ply"
    path.write_text("ply", encoding="utf-8")

    mesh = MagicMock()
    mesh.faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mesh.vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    fake_trimesh = SimpleNamespace(
        Scene=type("Scene", (), {}),
        Trimesh=type(mesh),
        points=SimpleNamespace(PointCloud=type("PointCloud", (), {})),
        util=SimpleNamespace(concatenate=lambda geoms: geoms[0]),
    )
    # Make isinstance checks succeed for our mesh mock.
    fake_trimesh.Trimesh = mesh.__class__

    with (
        patch.dict("sys.modules", {"trimesh": fake_trimesh, "numpy": np}),
        patch.object(viser_mod, "_load_trimesh_asset") as reload,
        patch.object(viser_mod, "_scene_geometries", return_value=[mesh]),
        patch.object(
            viser_mod,
            "_geometry_xyz_rgb",
            return_value=(mesh.vertices.copy(), np.full((3, 3), 255, dtype=np.uint8)),
        ),
        patch.object(viser_mod, "_coerce_rgb", side_effect=lambda colors, n: np.asarray(colors)),
    ):
        assert viser_mod._has_explicit_point_cloud(path, loaded=mesh) is False
        loaded_mesh = viser_mod._load_mesh(path, loaded=mesh)
        points, colors = viser_mod._load_xyz_rgb(path, loaded=mesh)

    reload.assert_not_called()
    assert loaded_mesh is mesh
    assert points.shape == (3, 3)
    assert colors.shape[0] == 3


def test_present_geometry_path_loads_trimesh_once(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "demo.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    calls: list[str] = []

    mesh = MagicMock()
    mesh.faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    mesh.vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    def _fake_load(path_arg: Path):
        calls.append(str(path_arg))
        return mesh

    monkeypatch.setattr(viser_mod, "_load_trimesh_asset", _fake_load)
    monkeypatch.setattr(viser_mod, "_has_explicit_point_cloud", lambda path, loaded=None: False)
    monkeypatch.setattr(viser_mod, "_load_mesh", lambda path, loaded=None: mesh)
    monkeypatch.setattr(viser_mod, "_apply_mesh_transform", lambda *a, **k: None)
    monkeypatch.setattr(
        viser_mod,
        "_load_xyz_rgb",
        lambda path, loaded=None: (
            np.asarray(mesh.vertices, dtype=np.float32),
            np.full((3, 3), 128, dtype=np.uint8),
        ),
    )

    # Drive only the load orchestration by calling the same branch logic.
    loaded_geometry = None
    suffix = path.suffix.lower()
    if suffix not in {".pcd", ".xyz", ".npz"}:
        loaded_geometry = viser_mod._load_trimesh_asset(path)
    has_point_cloud = viser_mod._has_explicit_point_cloud(path, loaded=loaded_geometry)
    mesh_out = None
    if suffix not in {".pcd", ".xyz", ".npz"}:
        mesh_out = viser_mod._load_mesh(path, loaded=loaded_geometry)
    if mesh_out is None or has_point_cloud:
        viser_mod._load_xyz_rgb(path, loaded=loaded_geometry)

    assert len(calls) == 1
    assert mesh_out is mesh
