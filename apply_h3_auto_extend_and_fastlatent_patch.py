from __future__ import annotations
import datetime as _dt
import py_compile
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / "payload"
FILES = [
    Path("iamccs_minimax_h3_shotboard.py"),
    Path("iamccs_minimax_h3_fast_latent_2pass.py"),
    Path("web") / "iamccs_h3_settings_pro_ui.js",
]


def resolve_target() -> Path:
    candidates = []
    if len(sys.argv) > 1 and sys.argv[1].strip():
        candidates.append(Path(sys.argv[1].strip().strip('"')))
    candidates += [ROOT, ROOT.parent, Path.cwd()]
    for cand in candidates:
        cand = cand.resolve()
        if all((cand / rel).exists() for rel in FILES):
            return cand
    raise SystemExit(
        "Could not locate IAMCCS-nodes automatically. Place this patch folder inside IAMCCS-nodes or pass the folder path as the first argument."
    )


def main() -> int:
    target = resolve_target()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Target IAMCCS-nodes: {target}")
    for rel in FILES:
        src = PAYLOAD / rel
        dst = target / rel
        if not src.exists():
            raise SystemExit(f"Missing payload file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            backup = dst.with_suffix(dst.suffix + f".bak.{stamp}")
            shutil.copy2(dst, backup)
            print(f"Backup: {backup.name}")
        shutil.copy2(src, dst)
        print(f"Patched: {rel}")

    for rel in [Path("iamccs_minimax_h3_shotboard.py"), Path("iamccs_minimax_h3_fast_latent_2pass.py")]:
        py_compile.compile(str(target / rel), doraise=True)
    print("\nPatch applied successfully.")
    print("Restart ComfyUI and hard-refresh the browser UI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
