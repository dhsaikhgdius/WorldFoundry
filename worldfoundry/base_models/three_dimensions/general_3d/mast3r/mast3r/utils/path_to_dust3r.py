# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dust3r sibling integration import
# --------------------------------------------------------

"""Module for base_models -> three_dimensions -> general_3d -> mast3r -> mast3r -> utils -> path_to_dust3r.py functionality."""

import sys
import os.path as path
HERE_PATH = path.normpath(path.dirname(__file__))
DUSt3R_REPO_PATH = path.normpath(path.join(HERE_PATH, '../../../dust3r'))
DUSt3R_LIB_PATH = path.join(DUSt3R_REPO_PATH, 'dust3r')
# check the presence of models directory in repo to be sure its cloned
if path.isdir(DUSt3R_LIB_PATH):
    # Modified by WorldFoundry: idempotent insert instead of remove+insert(0),
    # so repeated imports cannot reshuffle sys.path precedence.
    if DUSt3R_REPO_PATH not in sys.path:
        sys.path.insert(0, DUSt3R_REPO_PATH)
else:
    raise ImportError(f"dust3r is not initialized, could not find: {DUSt3R_LIB_PATH}.\n "
                      "Expected the canonical worldfoundry DUSt3R integration next to MASt3R.")
