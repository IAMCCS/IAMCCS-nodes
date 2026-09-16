"""Restore the IAMCCS files saved before the isolated Fused Turbo Preview branch.

Usage (from this folder):
    python rollback_fused_turbo_preview_20260902.py --dry-run
    python rollback_fused_turbo_preview_20260902.py --yes

The script restores only the four files changed by this feature. It never
touches workflows, models, outputs, settings JSON or unrelated IAMCCS code.
Restart ComfyUI after a real restore.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / ".codex_backups" / "fused_turbo_preview_20260902"
FILES = (
    Path("iamccs_minimax_h3_shotboard.py"),
    Path("iamccs_minimax_h3_shotboard_core.py"),
    Path("iamccs_minimax_h3_atomic_backend.py"),
    Path("web") / "iamccs_minimax_h3_shotboard_ui.js",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback IAMCCS Fused Turbo Preview only.")
    parser.add_argument("--dry-run", action="store_true", help="List files without changing anything.")
    parser.add_argument("--yes", action="store_true", help="Perform the restore.")
    args = parser.parse_args()
    if not BACKUP.is_dir():
        raise SystemExit(f"Backup is missing: {BACKUP}")
    if not args.dry_run and not args.yes:
        raise SystemExit("Refusing to overwrite files. Re-run with --dry-run or --yes.")
    for relative in FILES:
        source = BACKUP / relative.name
        destination = ROOT / relative
        if not source.is_file():
            raise SystemExit(f"Backup file is missing: {source}")
        print(f"{'WOULD RESTORE' if args.dry_run else 'RESTORING'} {relative}")
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    if not args.dry_run:
        print("Rollback complete. Restart ComfyUI to reload Python and JavaScript.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
