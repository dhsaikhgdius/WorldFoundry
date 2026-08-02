"""Regression coverage for the Uni3C stage-two Workspace contract."""

from worldfoundry.core.inference import get_model_inference_spec
from worldfoundry.studio.catalog import find_entry


def test_uni3c_exposes_required_stage_one_inputs_and_full_defaults() -> None:
    spec = get_model_inference_spec("uni3c")
    assert spec is not None
    task = spec.task()
    fields = {field.field_id: field for field in task.inputs}

    assert fields["input_path"].required is True
    assert fields["render_path"].required is True
    assert fields["render_path"].default is None
    assert fields["num_frames"].default == 81
    assert fields["fps"].default == 16
    assert fields["seed"].default == 1024
    assert fields["max_area"].default == 480 * 768


def test_uni3c_catalog_forwards_stage_two_runtime_inputs() -> None:
    entry = find_entry("uni3c")

    assert entry.default_task_type == "image-to-video"
    assert "render_path" in entry.call_params
    assert "controller_path" in entry.load_params
    assert "base_model_path" in entry.load_params
