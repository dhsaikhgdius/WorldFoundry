# DA-01 follow-up (not in this PR)

Tracked `worldfoundry/data/test_cases/` is already ~500 files / ~338 MiB and is
also listed in `.gitignore` (ignore rules do not untrack existing paths).

CI now pins that baseline via `scripts/ci/check_test_cases_growth.py` so PRs
cannot silently enlarge the checkout.

Deferred work (do **not** rewrite git history in the growth-gate PR):
- Migrate bulky fixtures to Git LFS and/or Hugging Face datasets.
- `git rm --cached` after migration, then shrink history in a coordinated cutover.
- Keep tiny smoke fixtures in-tree for CPU Studio/demo defaults.
