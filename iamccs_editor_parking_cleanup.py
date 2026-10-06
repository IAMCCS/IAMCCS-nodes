"""Conservative cleanup of one editor parking session."""
import json
import os
import time
from pathlib import Path


def purge_unused(session_dir, manifests, saved_roots):
    root = Path(session_dir).resolve(strict=True)
    protected = set()

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str):
            text = value.strip()
            if text.startswith(('{', '[')):
                try:
                    collect(json.loads(text))
                except ValueError:
                    pass
            else:
                protected.add(text.replace('\\', '/').rsplit('/', 1)[-1])

    if not isinstance(manifests, list) or not manifests:
        raise ValueError('Open editor manifests are required; reload the frontend.')
    for manifest in manifests:
        if not isinstance(manifest, dict) or manifest.get('schema') != 'iamccs.shotboard_video_editor.v1':
            raise ValueError('Invalid editor manifest; nothing deleted.')
        # Unused library assets are still retained intentionally. Remove them
        # from the project before cleanup, rather than guessing user intent.
        collect(manifest)
    scanned = []
    for location in saved_roots:
        location = Path(location)
        if not location.is_dir():
            continue
        scanned.append(str(location))
        for base, dirs, files in os.walk(location, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()
                       and not d.startswith('.') and d not in {'node_modules', '__pycache__'}]
            for name in files:
                if not name.lower().endswith('.json'):
                    continue
                path = Path(base) / name
                if path.is_symlink():
                    continue
                try:
                    text = path.read_text(encoding='utf-8-sig')
                    if 'parking' not in text.lower() and 'iamccs.shotboard_video_editor' not in text:
                        continue
                    collect(json.loads(text))
                except (OSError, ValueError) as exc:
                    raise ValueError(f'Cannot verify saved project {path}; nothing deleted.') from exc
    candidates = []
    retained = 0
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            continue
        # Only editor-generated take files and their previews/audio companions.
        if not path.name.startswith(('T', 'A')) or path.suffix.lower() not in {'.pt', '.png', '.jpg', '.mp4', '.wav'}:
            continue
        if path.name in protected or time.time() - path.stat().st_mtime < 3600:
            retained += 1
            continue
        candidates.append(path)
    deleted = total = 0
    failed = []
    for path in candidates:
        try:
            size = path.stat().st_size
            path.unlink()
            deleted += 1
            total += size
        except OSError as exc:
            failed.append({'path': str(path), 'error': str(exc)})
    return dict(ok=True, deleted_files=deleted, deleted_bytes=total,
                failed_files=len(failed), failed_paths=failed, retained_files=retained,
                folder=str(root), scanned_roots=scanned)
