import json
from pathlib import Path

import pytest

from worldfoundry.training.visual_generation import (
    UnsupportedWorldPlayModelError,
    build_training_plan,
    list_targets,
    resolve_stage,
    resolve_target,
)
from worldfoundry.training.visual_generation.assets import load_wan_config, normalize_config_paths
from worldfoundry.training.visual_generation.cli import main as cli_main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_visual_generation_training_targets_are_in_tree():
    targets = list_targets()
    assert {target.id for target in targets} == {"hy-action2v", "hy-ti2v", "wan-action2v"}

    for target in targets:
        assert target.components
        for component in target.components:
            assert component.import_path.startswith("worldfoundry.")
            assert "runtime/" not in component.import_path


def test_wan_training_uses_base_models_not_synthesis_wan_tree():
    assert (REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan").is_dir()
    assert (REPO_ROOT / "worldfoundry/base_models/diffusion_model/video/wan/wan_2p1/modules/action_model.py").is_file()
    assert not (REPO_ROOT / "worldfoundry/training/visual_generation/hunyuan_world/models/hyvideo").exists()
    assert not (REPO_ROOT / "worldfoundry/training/visual_generation/hunyuan_world/models/wan").exists()
    assert not (REPO_ROOT / "worldfoundry/training/visual_generation/wan/models").exists()
    assert (REPO_ROOT / "worldfoundry/training/visual_generation/wan/stage_models").is_dir()

    component_paths = {component.import_path for component in resolve_target("wan-action2v").components}
    assert all("worldfoundry.synthesis.visual_generation.wan" not in path for path in component_paths)
    assert any(
        path.startswith("worldfoundry.base_models.diffusion_model.video.wan")
        for path in component_paths
    )


def test_visual_generation_training_aliases_and_stage_aliases():
    target = resolve_target("tencent/HY-WorldPlay")
    assert target.id == "hy-action2v"
    assert resolve_stage(target, "stage0").id == "bidirectional-sft"
    assert resolve_stage(target, "tf").id == "ar-teacher-forcing"


def test_visual_generation_training_rejects_unverified_wan_inference_models():
    with pytest.raises(UnsupportedWorldPlayModelError):
        resolve_target("wan-t2v")


def test_visual_generation_training_plan_references_existing_worldfoundry_components():
    plan = build_training_plan(
        "hy-worldplay",
        "stage0",
        config_path="configs/visual_generation/train.yaml",
        output_dir="runs/visual_generation",
    )
    payload = plan.to_dict()

    assert payload["model_id"] == "hy-action2v"
    assert payload["stage_id"] == "bidirectional-sft"
    assert payload["command"][1:3] == [
        "-m",
        "torch.distributed.run",
    ]
    assert "worldfoundry.training.visual_generation.hunyuan_world.pipelines_camera.bidir_camera_training_entry" in payload[
        "command"
    ]
    assert payload["command"][payload["command"].index("--master_port") + 1] == "29613"
    assert payload["command"][payload["command"].index("--nproc_per_node") + 1] == "8"
    assert payload["command"][payload["command"].index("--sp-size") + 1] == "8"
    assert payload["command"][payload["command"].index("--cls-name") + 1] == "HunyuanTransformer3DARActionProPEModel"
    assert payload["command"][payload["command"].index("--json-path") + 1] == "dataset/HY15/Action2V/train_index.json"
    assert any(component["role"] == "pipeline" for component in payload["components"])
    assert all(component["import_path"].startswith("worldfoundry.") for component in payload["components"])
    assert payload["env_overrides"]["WANDB_MODE"] == "offline"
    assert payload["env_overrides"]["TOKENIZERS_PARALLELISM"] == "false"


def test_hunyuan_ti2v_training_plan_matches_minwm_parallel_defaults():
    plan = build_training_plan("hy-ti2v", "stage0")
    command = plan.command()

    assert command[1:3] == ("-m", "torch.distributed.run")
    assert "worldfoundry.training.visual_generation.hunyuan_world.pipelines.bidir_hunyuan_training_entry" in command
    assert command[command.index("--master_port") + 1] == "29612"
    assert command[command.index("--nproc_per_node") + 1] == "8"
    assert command[command.index("--num-gpus") + 1] == "8"
    assert command[command.index("--sp-size") + 1] == "2"
    assert command[command.index("--hsdp-shard-dim") + 1] == "8"
    assert command[command.index("--cls-name") + 1] == "HunyuanTransformer3DARActionModel"
    assert command[command.index("--json-path") + 1] == "dataset/HY15/TI2V/train_index.json"


def test_hunyuan_ode_sampling_uses_minwm_single_sp_default():
    command = build_training_plan("hy-ti2v", "ode-sampling").command()

    assert command[command.index("--master_port") + 1] == "29800"
    assert command[command.index("--sp-size") + 1] == "1"


def test_hunyuan_training_plan_distributed_values_can_come_from_env_override():
    plan = build_training_plan(
        "hy-action2v",
        "stage0",
        env_overrides={
            "WORLDFOUNDRY_VISUAL_GENERATION_NPROC_PER_NODE": "4",
            "WORLD_SIZE": "2",
            "RANK": "1",
            "MASTER_ADDR": "10.0.0.9",
            "MASTER_PORT": "29998",
            "SP_SIZE": "4",
        },
    )
    command = plan.command()

    assert command[command.index("--nproc_per_node") + 1] == "4"
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--node_rank") + 1] == "1"
    assert command[command.index("--master_addr") + 1] == "10.0.0.9"
    assert command[command.index("--master_port") + 1] == "29998"
    assert command[command.index("--num-gpus") + 1] == "8"
    assert command[command.index("--sp-size") + 1] == "4"
    assert command[command.index("--hsdp-shard-dim") + 1] == "8"


def test_wan_training_plan_uses_default_in_tree_config_and_entrypoint():
    plan = build_training_plan("wan-action2v", "stage1")
    payload = plan.to_dict()

    assert payload["stage_id"] == "ar-teacher-forcing"
    assert payload["command"][1:3] == ["-m", "torch.distributed.run"]
    assert payload["command"][payload["command"].index("--master_port") + 1] == "29601"
    assert payload["command"][payload["command"].index("--nproc_per_node") + 1] == "8"
    assert payload["command"][payload["command"].index("--nnodes") + 1] == "1"
    assert payload["command"][payload["command"].index("--node_rank") + 1] == "0"
    assert payload["command"][payload["command"].index("--sp_size") + 1] == "4"
    assert "-m" in payload["command"]
    assert "worldfoundry.training.visual_generation.wan.train" in payload["command"]
    assert "--config_path" in payload["command"]
    assert payload["config_path"].endswith(
        "worldfoundry/data/models/runtime/configs/wan_action2v/training/ar_camera_tf.yaml"
    )
    assert payload["env_overrides"]["WANDB_MODE"] == "offline"
    assert payload["env_overrides"]["NCCL_DEBUG"] == "WARN"
    assert any(component["role"] == "causal_backbone" for component in payload["components"])


def test_wan_training_plan_distributed_values_can_come_from_env_override(monkeypatch):
    monkeypatch.delenv("WORLDFOUNDRY_VISUAL_GENERATION_NPROC_PER_NODE", raising=False)
    plan = build_training_plan(
        "wan-action2v",
        "stage1",
        env_overrides={
            "WORLDFOUNDRY_VISUAL_GENERATION_NPROC_PER_NODE": "4",
            "WORLD_SIZE": "2",
            "RANK": "1",
            "MASTER_ADDR": "10.0.0.8",
            "MASTER_PORT": "29999",
            "SP_SIZE": "2",
        },
    )
    command = plan.command()

    assert command[command.index("--nproc_per_node") + 1] == "4"
    assert command[command.index("--nnodes") + 1] == "2"
    assert command[command.index("--node_rank") + 1] == "1"
    assert command[command.index("--master_addr") + 1] == "10.0.0.8"
    assert command[command.index("--master_port") + 1] == "29999"
    assert command[command.index("--sp_size") + 1] == "2"


def test_wan_training_plan_does_not_duplicate_explicit_sp_size():
    plan = build_training_plan("wan-action2v", "stage1", extra_args=("--sp_size", "1"))
    command = plan.command()

    assert command.count("--sp_size") == 1
    assert command[command.index("--sp_size") + 1] == "1"


def test_wan_lmdb_prep_stage_matches_minwm_torchrun_defaults(tmp_path: Path):
    plan = build_training_plan(
        "wan-action2v",
        "prepare-lmdb",
        output_dir=tmp_path / "Wan21" / "Action2V",
        env_overrides={
            "WORLDFOUNDRY_VISUAL_GENERATION_DATA_ROOT": str(tmp_path / "dataset"),
            "WORLDFOUNDRY_WAN_MODEL_ROOT": str(tmp_path / "ckpts"),
        },
    )
    payload = plan.to_dict()
    command = plan.command()

    assert payload["stage_id"] == "prepare-lmdb"
    assert payload["config_path"] is None
    assert command[1:3] == ("-m", "torch.distributed.run")
    assert command[command.index("--master_port") + 1] == "29700"
    assert command[command.index("--nproc_per_node") + 1] == "8"
    assert "worldfoundry.training.visual_generation.wan.utils.build_worldplaygen_lmdb" in command
    assert command[command.index("--input_json") + 1] == str(tmp_path / "dataset" / "preencode_input.json")
    assert command[command.index("--video_dir") + 1] == str(tmp_path / "dataset" / "videos")
    assert command[command.index("--output_dir") + 1] == str(tmp_path / "Wan21" / "Action2V")
    assert command[command.index("--vae_path") + 1] == str(
        tmp_path / "ckpts" / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"
    )


def test_wan_config_loader_accepts_minwm_fake_score_names(tmp_path: Path, monkeypatch):
    ckpt_root = tmp_path / "ckpts"
    monkeypatch.setenv("WORLDFOUNDRY_VISUAL_GENERATION_CKPT_ROOT", str(ckpt_root))
    config_path = tmp_path / "minwm_dmd.yaml"
    config_path.write_text(
        "\n".join(
            [
                "fake_ckpt: ./ckpts/Wan21/Action2V/bidirectional/model.pt",
                "fake_name: Wan2.1-T2V-1.3B",
                "fake_score_fsdp_wrap_strategy: size",
                "dfake_gen_update_ratio: 5",
            ]
        ),
        encoding="utf-8",
    )

    config = normalize_config_paths(load_wan_config(config_path), config_path=config_path)

    assert config["critic_ckpt"] == str(ckpt_root / "Wan21" / "Action2V" / "bidirectional" / "model.pt")
    assert config["critic_name"] == "Wan2.1-T2V-1.3B"
    assert config["critic_score_fsdp_wrap_strategy"] == "size"
    assert config["critic_gen_update_ratio"] == 5


def test_visual_generation_training_preflight_checks_module_specs():
    plan = build_training_plan("wan-action2v", "causal-ode")
    checks = plan.preflight()
    assert checks
    assert all(check.mode == "module-spec" for check in checks)
    assert all(check.ok for check in checks)


def test_visual_generation_training_cli_plan_json(capsys):
    assert cli_main(["plan", "--model", "hunyuan-ti2v", "--stage", "stage3", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_id"] == "hy-ti2v"
    assert payload["stage_id"] == "dmd"
    assert payload["components"]
