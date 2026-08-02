from __future__ import annotations

import pytest

from worldfoundry.studio.catalog import STUDIO_HIDDEN_CATALOG_MODEL_IDS, discover_catalog, find_entry
from worldfoundry.studio.studio_catalog import _template_id_hint


_TEMPLATE_TO_WORKLOAD = {
    "depth-geometry": "geometry",
    "scene-3d": "3d",
    "hosted-api": "api",
    "conditioned-video": "i2v",
    "video-to-video": "v2v",
}


@pytest.mark.parametrize(
    ("model_id", "category", "template_id", "workload"),
    [
        ("dust3r", "Depth / Geometry", "depth-geometry", "geometry"),
        ("dust3r-base-model", "Depth / Geometry", "depth-geometry", "geometry"),
        ("inspatio-world", "Video-to-Video", "video-to-video", "v2v"),
        ("recammaster", "Video-to-Video", "video-to-video", "v2v"),
        ("neoverse", "Video-to-Video", "video-to-video", "v2v"),
        ("worldlabs-marble-1.1", "Remote API", "hosted-api", "api"),
        ("dvlt", "3D Scene", "scene-3d", "3d"),
        ("lagernvs", "3D Scene", "scene-3d", "3d"),
        ("lingbot-map", "3D Scene", "scene-3d", "3d"),
        ("lyra-1", "3D Scene", "scene-3d", "3d"),
        ("lyra-2", "3D Scene", "scene-3d", "3d"),
    ],
)
def test_studio_catalog_model_taxonomy(model_id: str, category: str, template_id: str, workload: str) -> None:
    entry = find_entry(model_id)
    resolved_template = _template_id_hint(entry)

    assert entry.category == category
    assert resolved_template == template_id
    assert _TEMPLATE_TO_WORKLOAD.get(resolved_template, "world") == workload


@pytest.mark.parametrize("model_id", sorted(STUDIO_HIDDEN_CATALOG_MODEL_IDS))
def test_unintegrated_or_duplicate_models_are_hidden_from_studio(model_id: str) -> None:
    studio_ids = {entry.model_id for entry in discover_catalog()}

    assert model_id not in studio_ids
