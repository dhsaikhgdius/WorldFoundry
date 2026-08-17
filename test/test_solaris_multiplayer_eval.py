from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

# HANDOVER(module owners): worldfoundry.evaluation.tasks.official.solaris_multiplayer
# was removed together with the tasks.official package and has no replacement
# under evaluation.tasks.execution. The solaris pipeline/synthesis still exist,
# so this suite needs a rewrite against the current eval task surface (if the
# multiplayer eval flow still has one). Skipped so `pytest test` can collect.
# See plan/code_review/fixes/14_tests_ci_fixes.md.
pytest.skip(
    "worldfoundry.evaluation.tasks.official.solaris_multiplayer was removed; "
    "suite needs rewrite against the current eval task surface",
    allow_module_level=True,
)

import worldfoundry.evaluation.tasks.official.solaris_multiplayer as solaris_task_module


class _DummySolarisPipe:
    def __init__(self) -> None:
        self.calls = []
        self.synthesis_model = type(
            "Synthesis",
            (),
            {
                "runtime_root": "/tmp/solaris",
                "eval_data_dir": "/tmp/solaris/datasets",
                "eval_python_executable": "/tmp/solaris/bin/python",
            },
        )()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "model_output_dir": "/tmp/out/demo",
            "generated_videos": {
                "translation": ["/tmp/out/demo/eval_translation/video_0_side_by_side.mp4"]
            },
            "selected_eval_types": ["translation"],
            "experiment_name": "demo",
            "runtime_root": "/tmp/solaris",
        }


def _create_fake_runtime_root(tmp_path: Path) -> Path:
    runtime_root = tmp_path / "solaris"
    (runtime_root / "src").mkdir(parents=True)
    (runtime_root / "config").mkdir(parents=True)
    (runtime_root / "vlm_eval").mkdir(parents=True)
    (runtime_root / "src" / "inference.py").write_text("# stub\n", encoding="utf-8")
    (runtime_root / "config" / "inference.yaml").write_text("defaults: []\n", encoding="utf-8")
    (runtime_root / "vlm_eval" / "run_eval.py").write_text("# stub\n", encoding="utf-8")
    return runtime_root


def test_solaris_multiplayer_reference_func_calls_pipeline_directly(tmp_path: Path) -> None:
    pipe = _DummySolarisPipe()
    output_hint = tmp_path / "results" / "video_dir" / "sample.mp4"

    payload = solaris_task_module.reference_func(
        pipe,
        pipe_infer=None,
        input_data_info={
            "sample_id": "demo",
            "eval_types": ["translation"],
            "output_path": str(output_hint),
            "eval_num_samples": 2,
        },
        output_key="generated_video_dir",
    )

    assert payload["generated_video_dir"] == "/tmp/out/demo"
    assert payload["runtime_root"] == "/tmp/solaris"
    assert payload["eval_data_dir"] == "/tmp/solaris/datasets"
    assert payload["eval_python_executable"] == "/tmp/solaris/bin/python"
    assert pipe.calls
    assert pipe.calls[0]["output_dir"] == str(output_hint.parent)
    assert pipe.calls[0]["eval_num_samples"] == 2
    assert pipe.calls[0]["return_dict"] is True


def test_solaris_multiplayer_eval_func_runs_vlm_eval_and_reads_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    runtime_root = _create_fake_runtime_root(tmp_path)
    generated_dir = tmp_path / "generated" / "demo"
    generated_dir.mkdir(parents=True)
    eval_data_dir = tmp_path / "datasets"
    (eval_data_dir / "eval" / "translationEval").mkdir(parents=True)

    calls = []

    def fake_run(command, check, cwd, env, stdout, stderr):
        calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": env,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        results_dir = Path(command[command.index("--results-dir") + 1])
        output_dir = results_dir / "generated" / f"{generated_dir.name}_translationEval"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "stats.json").write_text(
            json.dumps(
                {
                    "metric": "episode_level_accuracy.episode_accuracy",
                    "num_trials": 1,
                    "mean": 87.5,
                    "median": 87.5,
                    "std": 0.0,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(solaris_task_module.subprocess, "run", fake_run)

    result = solaris_task_module.eval_func(
        input_data_info={
            "sample_id": "demo",
            "track": "solaris_multiplayer_vlm_eval",
            "generated_video_dir_path": str(generated_dir),
            "runtime_root": str(runtime_root),
            "eval_data_dir": str(eval_data_dir),
            "eval_python_executable": "/tmp/solaris/bin/python",
            "selected_eval_types": ["translation"],
            "api_key": "test-api-key",
            "num_trials": 1,
            "limit": 2,
        },
        eval_pipeline=None,
        eval_pipeline_infer=None,
    )

    assert calls
    assert calls[0]["command"][:2] == ["/tmp/solaris/bin/python", "run_eval.py"]
    assert "--limit" in calls[0]["command"]
    assert calls[0]["cwd"] == str(runtime_root / "vlm_eval")
    assert calls[0]["env"]["GEMINI_API_KEY"] == "test-api-key"
    assert result["episode_accuracy_mean"] == 87.5
    assert result["translation_episode_accuracy"] == 87.5
    assert result["successful_eval_types"] == 1
    assert result["per_eval"]["translation"]["episode_accuracy_mean"] == 87.5
