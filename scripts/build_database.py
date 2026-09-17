"""
Full database build: labels + events for all GAGES-II gages.

Reads daily streamflow from the local CSV cache in data/raw/nwis/.

Usage:
    python scripts/build_database.py
    python scripts/build_database.py --reference-only   # phase 1 build
    python scripts/build_database.py --force            # reprocess all
"""

import click
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bfd_db.config import PipelineConfig
from bfd_db.data.gagesii import load_gagesii_metadata, get_reference_gages, get_gage_ids
from bfd_db.database.build import build_database


@click.command()
@click.option("--reference-only", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--t-c", default=None, type=int, help="Override t_c (gap tolerance days).")
@click.option("--d-min", default=None, type=int, help="Override d_min (min event days).")
@click.option("--x-pct", default=None, type=float, help="Override BF ratio threshold (0–1).")
def main(reference_only, force, t_c, d_min, x_pct):
    config = PipelineConfig()
    if t_c is not None:
        config.t_c = t_c
    if d_min is not None:
        config.d_min = d_min
    if x_pct is not None:
        config.bf_ratio_threshold = x_pct

    meta = load_gagesii_metadata()
    if reference_only:
        gage_ids = get_reference_gages(meta)
    else:
        gage_ids = None  # uses full GAGES-II set

    build_database(config=config, gage_ids=gage_ids, force=force)


if __name__ == "__main__":
    main()
