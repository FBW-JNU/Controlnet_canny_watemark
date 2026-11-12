"""Restore the archived population manifest under a new local COCO root.

Standard library only. Preserves row order, prompts and all statistics; changes
image_path only. Does not download images or claim old absolute-path hashes.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coco-root', type=Path, required=True,
                        help='Directory containing train2017/')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--check-images', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'data/coco/normal_manifest.csv.gz'
    metadata = json.loads((ROOT / 'data/coco/manifest_metadata.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != metadata['portable_gzip_sha256']:
        raise SystemExit('Archived manifest hash mismatch.')
    if args.out.exists():
        raise SystemExit('Output exists; choose a new path.')
    with gzip.open(source, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != metadata['rows'] or fields != metadata['columns']:
        raise SystemExit('Manifest schema/count mismatch.')
    coco_root = args.coco_root.resolve()
    missing = []
    for row in rows:
        filename = row['file_name']
        if Path(filename).name != filename or row['image_path'] != 'train2017/' + filename:
            raise SystemExit('Unexpected image path in archived manifest.')
        image = coco_root / 'train2017' / filename
        if args.check_images and not image.is_file():
            missing.append(str(image))
        row['image_path'] = str(image)
    if missing:
        raise SystemExit(f'{len(missing)} images missing; first: {missing[0]}; no CSV written.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(rows=len(rows),output=str(args.out.resolve()),
        sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(),
        image_existence='checked' if args.check_images else 'not checked',
        note='Migrated image paths; record a new protocol hash.'),indent=2))


if __name__ == '__main__':
    main()
