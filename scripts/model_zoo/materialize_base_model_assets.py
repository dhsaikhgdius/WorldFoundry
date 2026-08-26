#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

from worldfoundry.core.io.paths import project_root
from worldfoundry.core.process import run_logged_subprocess

REPO_ROOT = project_root(__file__)
SRC_ROOT = REPO_ROOT
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

CAPABILITIES_PATH = SRC_ROOT / "worldfoundry" / "base_models" / "capabilities.py"
CAPABILITIES_SPEC = importlib.util.spec_from_file_location("worldfoundry_base_model_capabilities", CAPABILITIES_PATH)
if CAPABILITIES_SPEC is None or CAPABILITIES_SPEC.loader is None:
    raise RuntimeError(f"cannot load base-model capabilities from {CAPABILITIES_PATH}")
CAPABILITIES = importlib.util.module_from_spec(CAPABILITIES_SPEC)
sys.modules[CAPABILITIES_SPEC.name] = CAPABILITIES
CAPABILITIES_SPEC.loader.exec_module(CAPABILITIES)

BASE_MODEL_CAPABILITIES = CAPABILITIES.BASE_MODEL_CAPABILITIES
BASE_MODEL_STACKS = CAPABILITIES.BASE_MODEL_STACKS
base_model_inventory = CAPABILITIES.base_model_inventory
base_model_materialization_plan = CAPABILITIES.base_model_materialization_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or download reusable WorldFoundry base-model assets.")
    parser.add_argument(
        "--capability",
        action="append",
        choices=sorted([*BASE_MODEL_CAPABILITIES, *BASE_MODEL_STACKS]),
        help="Capability or stack id to materialize. May repeat. Defaults to all registered capabilities.",
    )
    parser.add_argument("--list", action="store_true", help="List registered base-model capabilities and stacks.")
    parser.add_argument("--execute-downloads", action="store_true", help="Run generated hf download commands.")
    parser.add_argument("--json", action="store_true", help="Print the full machine-readable plan.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        inventory = base_model_inventory()
        if args.json:
            print(json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True))
        else:
            print(f"base-model capabilities: {inventory['capability_count']}")
            for item in inventory["capabilities"]:
                print(f"capability {item['id']} [{item['family']}]")
            print(f"base-model stacks: {inventory['stack_count']}")
            for item in inventory["stacks"]:
                print(f"stack {item['id']} [{item['family']}] -> {', '.join(item['capability_ids'])}")
        return 0

    plan = base_model_materialization_plan(args.capability)
    executed = []
    if args.execute_downloads:
        log_root = REPO_ROOT / "tmp" / "model_zoo" / "base_model_download_logs"
        for index, command in enumerate(plan["download_command_argvs"]):
            log_dir = log_root / f"cmd-{index:03d}"
            stdout_path = log_dir / "download.stdout.log"
            stderr_path = log_dir / "download.stderr.log"
            completed = run_logged_subprocess(
                [str(item) for item in command],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            stdout_text = (
                stdout_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if stdout_path.is_file()
                else ""
            )
            stderr_text = (
                stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:]
                if stderr_path.is_file()
                else ""
            )
            executed.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                }
            )
        plan["executed_downloads"] = executed
        plan = base_model_materialization_plan(args.capability)
        plan["executed_downloads"] = executed

    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("base-model assets:", "ok" if plan["ok"] else "missing")
        if plan["stack_ids"]:
            print("stacks:", ", ".join(plan["stack_ids"]))
        print("capabilities:", ", ".join(plan["capability_ids"]))
        if plan["pip_install_packages"]:
            print("install:", "python -m pip install " + " ".join(plan["pip_install_packages"]))
        for command in plan["download_commands"]:
            print("download:", command)
        for command in plan["export_commands"]:
            print("env:", command)
        for action in plan["manual_actions"]:
            print("manual:", action)
    return 0 if plan["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
