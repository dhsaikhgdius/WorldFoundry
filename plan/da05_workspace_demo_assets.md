# DA-05: Workspace demo asset pins

## Official git sources

`scripts/setup/workspace_demo_asset_pins.yaml` is the single source of truth for
demo inputs copied from upstream checkouts. Each `repo_assets` entry resolves
through a `repos.<name>.revision` (never `HEAD`). Historical paths that were
later deleted upstream keep an explicit per-asset `revision`.

```bash
python scripts/setup/materialize_workspace_demo_assets.py --check --json
python scripts/setup/materialize_workspace_demo_assets.py --force
```

When an asset pin sets `expected_sha256`, materialize/check marks
`sha256_mismatch` if the on-disk tree differs. After a trusted materialize,
record hashes into the YAML and optionally gate with `--require-sha256`.

## EXTERNAL dataset assets (manual)

These remain `external_pending` until staged under `worldfoundry/data/test_cases/`:

| Target | Source URI |
| --- | --- |
| `multiworld_ittakestwo/action.csv` | `hf://datasets/Haoyuwu/MultiWorldData/480P_eval_chunk0001.tar#000100_f564185_564266.csv` |
| `multiworld_ittakestwo/input.png` | same tar, MP4 frame 0 |
| `test_vla_case1/aloha/observation_images_cam_*.png` | `hf://datasets/lerobot/aloha_static_vinh_cup/...` frame 0 |
| `test_vla_image_case1/init_frame.png` | same LeRobot cam_high frame 0 |

Suggested manual flow:

1. Download the Hugging Face dataset shard / video with the Hub CLI or browser.
2. Extract the CSV / decode frame 0 to PNG.
3. Place files at the targets above (do **not** grow tracked `test_cases` blobs; keep them local/gitignored).
4. Re-run `--check` until status is `ready`.

Checkpoint-backed VLA sample images still come from `--ckpt-root` (see
`checkpoint_assets` in the pin YAML).
