# us_baseflow_db
National baseflow database

## Installation

```bash
pip install -e ".[labeling,database,analysis]"   # everything, for working on this repo
pip install -e ".[labeling]"                     # just the labeling/ensemble algorithm
```

`labeling` (baseflowx, pybfs, scikit-learn, joblib) is what downstream
consumers of the ensemble algorithm need. `database` (geopandas, pyarrow,
tqdm, click) is for `scripts/build_database.py`. `analysis` (matplotlib) is
for `bfd_db.analysis`.

## Downstream consumers

`low_flow_analyst` depends on `bfd-db[labeling]` (as a git dependency in its
`requirements.txt`) to run the same BFD ensemble algorithm — RF-BFD, the
baseflowx filters, PyBFS, the magnitude-adaptive threshold — that
`scripts/build_database.py` uses to build the frozen national snapshot, so a
gage's live demo in that app and its row in the database come from the same
code path. See `docs/adaptive-bfd-threshold.md` for the algorithm and
`low_flow_analyst`'s own `docs/adaptive-bfd-threshold.md` for how that app
wires it into its API and UI.

A change to `bfd_db.labeling` or `bfd_db.events` — new voter, different
threshold rule, different event pooling — is a change to what both projects
show. Add a test in `tests/`, then bump `low_flow_analyst`'s pinned commit in
its `requirements.txt` to pick it up (see that repo's doc, §4).
