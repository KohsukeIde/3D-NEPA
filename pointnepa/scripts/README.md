# pointNEPA Scripts

PointGPT / pointNEPA launchers live here so they do not appear as global NEPA
scripts.

## Layout

- `pointnepa/scripts/local/`
  - maintained local workstation entrypoints.
- `pointnepa/scripts/sanity/`
  - QF, smoke, one-off, and compatibility launchers.

Scripts default to the sibling upstream checkout:

```bash
POINTGPT_DIR="${WORKDIR}/PointGPT"
```

Run them from the repo root unless a script says otherwise.
