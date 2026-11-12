"""Validate CPU integration, seal a manifest and build a GitHub-sized ZIP.

Run from a complete exported folder with its CPU environment. This does not
publish anything and never invokes a model/generation/training command.
"""
from pathlib import Path
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SKIP={'__pycache__','.git','runs','.pytest_cache'}


def included(p):
    return p.is_file() and not any(t in SKIP for t in p.relative_to(ROOT).parts)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    runs=[]
    for command in [[sys.executable,'tools/smoke_check.py'],
                    [sys.executable,'tools/check_stability.py'],
                    [sys.executable,'tools/check_benchmark.py'],
                    [sys.executable,'tools/check_environment.py','--generation']]:
        r=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,check=True)
        runs.append(dict(command=' '.join(['python']+command[1:]),
                         exit_code=r.returncode,output=json.loads(r.stdout)))
    count=0
    for p in ROOT.rglob('*.py'):
        if included(p):
            compile(p.read_text(encoding='utf-8-sig'),str(p),'exec');count+=1
    for p in [ROOT/'README.md',ROOT/'THIRD_PARTY_NOTICES.md',*list((ROOT/'docs').glob('*.md'))]:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if not link.startswith(('http','#')) and not (p.parent/link.split('#')[0]).exists():
                raise RuntimeError(f'Broken doc link: {p.name}: {link}')
    snapshot=json.loads((ROOT/'provenance/source_manifest.json').read_text())
    for r in snapshot['records']:
        if 'destination' in r:
            assert sha(ROOT/r['destination'])==r['exported_sha256'],r['destination']
    validation=dict(date='2026-11-12',status='passed',python=platform.python_version(),
        checks=runs,python_files_compiled=count,documentation_links='passed',
        exported_source_hashes='passed',
        independent_directory_runtime=json.loads((ROOT/'provenance/portable_snapshot_validation.json').read_text())['status'],
        GPU_generation=json.loads((ROOT/'provenance/full_reproduction.json').read_text())['status'],
        GPU_scope='Historical748 PNGs and two independent288-image region-first runs; see full_reproduction.json',training='No neural training; historical4A and new Cal channel models reconstructed',
        fresh_environment_install='not run; local compatible environment used')
    (ROOT/'provenance/validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
           if included(p) and p!=ROOT/'provenance/bundle_files.json'}
    (ROOT/'provenance/bundle_files.json').write_text(json.dumps(files,indent=2)+'\n')
    subprocess.run([sys.executable,'tools/verify_bundle.py'],cwd=ROOT,env=env,check=True)
    archive=ROOT.parent/(ROOT.name+'_repro_20261112.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(ROOT.rglob('*')):
            if included(p):z.write(p,arcname=ROOT.name+'/'+str(p.relative_to(ROOT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(directory=ROOT.name,files=len(files)+1,uncompressed_bytes=sum(
        p.stat().st_size for p in ROOT.rglob('*') if included(p)),archive=archive.name,
        archive_bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))


if __name__=='__main__':
    main()
