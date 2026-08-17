# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# CroCo submodule import
# --------------------------------------------------------

"""Module for base_models -> three_dimensions -> general_3d -> stable_virtual_camera -> stable_virtual_camera_runtime -> third_party -> dust3r -> dust3r -> utils -> path_to_croco.py functionality."""

import os.path as path
import sys

HERE_PATH = path.normpath(path.dirname(__file__))
CROCO_REPO_PATH = path.normpath(path.join(HERE_PATH, "../../croco"))
CROCO_MODELS_PATH = path.join(CROCO_REPO_PATH, "models")
# check the presence of models directory in repo to be sure its cloned
if path.isdir(CROCO_MODELS_PATH):
    # workaround for sibling import
    # Modified by WorldFoundry: idempotent insert (avoid duplicate sys.path entries).
    if CROCO_REPO_PATH not in sys.path:
        sys.path.insert(0, CROCO_REPO_PATH)
else:
    raise ImportError(
        f"croco is not initialized, could not find: {CROCO_MODELS_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )
