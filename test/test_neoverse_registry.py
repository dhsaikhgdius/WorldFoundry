from pathlib import Path

from PIL import Image

from worldfoundry.operators.neoverse_operator import NeoVerseOperator
from worldfoundry.pipelines.neoverse.pipeline_neoverse import NeoVersePipeline


def test_neoverse_operator_video_and_image_perception():
    assert NeoVersePipeline.__name__ == "NeoVersePipeline"

    operator = NeoVerseOperator(frames_per_action=20)
    operator.get_interaction(["forward", "camera_left", "right"])
    processed = operator.process_interaction()
    operator.delete_last_interaction()

    assert processed["num_frames"] == 61
    assert processed["keyframes"][0] == {0: [{"static": {}}]}
    assert processed["actions"] == ["forward", "camera_l", "right"]

    image = Image.new("RGB", (640, 384), color="white")
    perception = operator.process_perception(image)
    assert perception["input_frames"][0].size == (560, 336)
    assert perception["static_scene"] is True

    demo_video = (
        Path(__file__).resolve().parents[1]
        / "worldfoundry/data/test_cases/neoverse/videos/robot.mp4"
    )
    video_perception = operator.process_perception(
        str(demo_video),
        height=64,
        width=96,
        num_frames=3,
    )
    assert len(video_perception["input_frames"]) == 3
    assert video_perception["input_frames"][0].size == (96, 64)
    assert video_perception["static_scene"] is False


class _FakeNeoVerseOperator:
    zoom_ratio = 1.0
    trajectory_mode = "relative"

    def __init__(self):
        self.perception_call = None

    def process_perception(self, images, **kwargs):
        self.perception_call = {"images": images, "kwargs": kwargs}
        return {
            "input_frames": [Image.new("RGB", (96, 64), color="white")],
            "static_scene": kwargs["static_scene"],
        }

    def get_interaction(self, interaction):
        self.interaction = interaction

    def process_interaction(self):
        return {
            "actions": [],
            "predefined_trajectory": "tilt_up",
            "trajectory_file": None,
            "trajectory_data": None,
            "keyframes": None,
            "num_frames": 81,
            "trajectory_mode": "relative",
            "trajectory_name": "tilt_up",
            "zoom_ratio": 1.0,
            "angle": 15,
            "distance": 0,
            "orbit_radius": 0,
            "use_first_frame": True,
        }

    def delete_last_interaction(self):
        pass


class _FakeNeoVerseSynthesis:
    height = 64
    width = 96

    def __init__(self):
        self.predict_call = None

    def predict(self, **kwargs):
        self.predict_call = kwargs
        return {"video": ["frame"]}


def test_neoverse_pipeline_accepts_workspace_video_alias_without_forcing_static_scene():
    operator = _FakeNeoVerseOperator()
    synthesis = _FakeNeoVerseSynthesis()
    pipeline = NeoVersePipeline(operator=operator, synthesis_model=synthesis)

    result = pipeline(
        video_path="/tmp/robot.mp4",
        predefined_trajectory="tilt_up",
        num_frames=81,
        use_first_frame=True,
        static_scene=False,
        return_dict=True,
    )

    assert result["video"] == ["frame"]
    assert operator.perception_call["images"] == "/tmp/robot.mp4"
    assert operator.perception_call["kwargs"]["num_frames"] == 81
    assert operator.perception_call["kwargs"]["static_scene"] is False
    assert synthesis.predict_call["static_scene"] is False
    assert synthesis.predict_call["use_first_frame"] is True
