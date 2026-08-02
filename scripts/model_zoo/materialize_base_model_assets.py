#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
        for command in plan["download_command_argvs"]:
            completed = subprocess.run([str(item) for item in command], text=True, capture_output=True, check=False)
            executed.append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
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
