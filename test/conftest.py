from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT

for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


# ---------------------------------------------------------------------------
# Manual demo scripts excluded from pytest collection.
#
# The top-level test_<model>.py files listed below contain no test functions:
# they are module-level demo/inference scripts. Importing them (which pytest
# does during collection) would trigger multi-GB checkpoint downloads, CUDA
# inference, real API calls, and write output files into the repository
# working tree.
#
# They are kept as manual smoke scripts. Run one directly instead of through
# pytest, e.g.:
#
#   PYTHONPATH=. python test/test_wan_2p2.py
#
# Do NOT add files with real pytest test functions to this list.
# ---------------------------------------------------------------------------
collect_ignore = [
    "test_astra.py",
    "test_cosmos_predict2p5.py",
    "test_cut3r.py",
    "test_depth_anything_v1.py",
    "test_depth_anything_v2.py",
    "test_depth_anything_v2_registry.py",
    "test_depth_anything_v3.py",
    "test_depth_anything_v3_registry.py",
    "test_fantasy_world.py",
    "test_fantasy_world_wan22_official_parity.py",
    "test_flash_world.py",
    "test_gen3c.py",
    "test_hailuo_2p3.py",
    "test_hunyuan_gamecraft.py",
    "test_hunyuan_mirror.py",
    "test_hunyuan_worldplay.py",
    "test_hunyuan_world_voyager.py",
    "test_hy_world_2p0.py",
    "test_hy_world_2p0_registry.py",
    "test_infinite_vggt.py",
    "test_infinite_world.py",
    "test_kling_api.py",
    "test_lingbot_world.py",
    "test_loger.py",
    "test_luma_ray2.py",
    "test_matrix_game_2.py",
    "test_matrix_game_3.py",
    "test_neoverse.py",
    "test_pi3.py",
    "test_recammaster.py",
    "test_runway_gen4p5.py",
    "test_sora2.py",
    "test_veo3.py",
    "test_vggt.py",
    "test_vggt_omega.py",
    "test_video_models.py",
    "test_vmem.py",
    "test_wan_2p2.py",
    "test_wan_2p5.py",
    "test_wan_2p6.py",
    "test_wan_2p7.py",
    "test_worldlabs.py",
    "test_yume.py",
    "test_yume_1p5.py",
]
