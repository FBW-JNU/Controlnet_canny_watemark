# 用精简版更新已有仓库

仅覆盖上传新文件不会删除旧文件。最稳妥方式是在**你自己的仓库clone目录**中先删除本次明确淘汰的旧文件，再复制新版、检查差异并正常提交；不需要force push。

本次完整删除清单在`provenance/simplification.json`的`removed_paths`。其中包含`research_snapshot/`、`vendor/controlnet/controlnet_robin/`和无关训练/诊断文件。原MAT Release不受影响。

以下脚本在仓库根目录运行，只删除清单中的旧文件。先复制新版`provenance/simplification.json`进入clone，再执行；应先提交或备份自己的未提交修改。

```python
from pathlib import Path
import json
root = Path.cwd()
manifest = json.loads((root / 'provenance/simplification.json').read_text())
for relative in manifest['removed_paths']:
    path = root / relative
    if path.is_file():
        print('remove:', relative)
        path.unlink()
```

之后将新版文件夹内容复制到clone根目录（不要再套一层目录），执行：

```bash
git status --short
git diff --stat
git add -A
git commit -m 'Simplify ControlNet backend and screen stability before curve extraction'
git push
```

若坚持网页分批上传，需在GitHub上手动删除清单中的旧目录/文件；新版文件上传后检查顶层仍有`canny_geometry`、`vendor`、`configs`等代码目录。代码ZIP不应解压进MAT Release；MAT Release继续保留独立权重ZIP。
