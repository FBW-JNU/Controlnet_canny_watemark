"""Import/version check only; no weight download, model load or GPU allocation."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import platform

p=argparse.ArgumentParser();p.add_argument('--generation',action='store_true');args=p.parse_args()
modules=['numpy','scipy','cv2','PIL']
if args.generation:modules+=['torch','torchvision','pytorch_lightning','transformers','open_clip','omegaconf','einops']
report={'python':platform.python_version(),'modules':{}}
failed=False
for name in modules:
    try:
        mod=importlib.import_module(name)
        report['modules'][name]={'status':'ok','version':getattr(mod,'__version__','unknown')}
    except Exception as exc:
        failed=True;report['modules'][name]={'status':'error','error':str(exc)}
print(json.dumps(report,indent=2))
raise SystemExit(1 if failed else 0)
