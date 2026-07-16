# Documentation

- `images/` — curated result figures used by the root `README.md`
- `presentation_2023.08.18/` — early method presentation and draft script
- `钢筋绑扎机器人及钢筋交叉点的识别方法.PDF` — related method notes
- `result_snapshots/` — raw experiment screenshots (local; ignored by git)

Regenerate matplotlib demo figures:

```bash
conda activate rebar_intersection
PYTHONPATH=. python scripts/generate_readme_figures.py
```
