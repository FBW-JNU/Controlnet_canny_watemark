"""Verify release hashes; excluded assets are intentional, not missing files."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'provenance/bundle_files.json').read_text())
for name,expected in manifest.items():
    p=ROOT/name
    if not p.is_file():raise SystemExit(f'Missing: {name}')
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise SystemExit(f'Hash mismatch: {name}')
for p in ROOT.rglob('*'):
    relative=p.relative_to(ROOT)
    if any(x in relative.parts for x in ['.git','__pycache__','runs']):continue
    if p.is_symlink():raise SystemExit(f'Unexpected symlink: {relative}')
    if p.is_file() and (p.stat().st_size>50*1024*1024 or p.suffix.lower() in ['.pth','.pt','.ckpt','.safetensors','.pkl']):
        raise SystemExit(f'Unexpected large/model asset: {relative}')
print(f'Verified {len(manifest)} release files; no bundled checkpoints or large assets.')
