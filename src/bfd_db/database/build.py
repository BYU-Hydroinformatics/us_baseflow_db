"""
Full pipeline orchestration: from GAGES-II + NWIS → daily labels → events → DB.

Call build_database() with a PipelineConfig to regenerate the national database
from scratch. Daily streamflow is read from the local CSV cache in
data/raw/nwis/ (see bfd_db.data.nwis). The pipeline is designed to be
resumable: gages whose label files already exist in data/processed/labels/
are skipped unless force=True.

Build order:
  Phase 1 — reference gages first (2,057 gages; validates ensemble quality)
  Phase 2 — full GAGES-II set (additional 7,265 non-reference gages)
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bfd_db.config import DATA_OUTPUTS, DATA_PROCESSED, PipelineConfig
from bfd_db.data.gagesii import load_gagesii_metadata, get_gage_ids, get_reference_gages
from bfd_db.data.nwis import load_daily_q
from bfd_db.labeling.ensemble import label_gage_all_methods
from bfd_db.labeling.rf_bfd import load_rf_bfd_model
from bfd_db.events.pooling import pool_events
from bfd_db.events.statistics import add_event_stats
from bfd_db.database.schema import create_provenance_record

LABELS_CACHE = DATA_PROCESSED / "labels"
LABELS_CACHE.mkdir(parents=True, exist_ok=True)


def _process_one_gage(
    site_no: str,
    config: PipelineConfig,
    meta: pd.DataFrame,
    rf_model,
    rf_scaler,
    force: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run the full pipeline for a single gage.

    Returns (daily_labels_df, events_df) or (None, None) on failure.
    """
    label_path = LABELS_CACHE / f"{site_no}.parquet"
    if label_path.exists() and not force:
        labels_df = pd.read_parquet(label_path)
    else:
        q_df = load_daily_q(site_no, config.start_date, config.end_date)
        if q_df.empty:
            return None, None
        q = q_df["Q_cfs"]

        label_results = label_gage_all_methods(
            site_no, q, config, rf_model, rf_scaler
        )
        labels_df = q_df.join(label_results)
        labels_df.insert(0, "STAID", site_no)
        labels_df.to_parquet(label_path)

    # Generate events from ensemble column
    events = pool_events(labels_df["ensemble"], config.t_c, config.d_min)
    if events.empty:
        return labels_df, events

    q_full = labels_df["Q_cfs"]
    events = add_event_stats(events, q_full)
    events.insert(0, "STAID", site_no)

    # Propagate provisional flag and GAGES-II metadata
    events["has_provisional"] = events.apply(
        lambda row: (
            labels_df.loc[row["start_date"] : row["end_date"], "approval"]
            .ne("A")
            .any()
        ),
        axis=1,
    )
    if site_no in meta.index:
        events["CLASS"] = meta.loc[site_no, "CLASS"]
        # HYDROmod_QAchange requires the full GAGES-II basinchar file, not
        # yet available — see bfd_db.data.gagesii module docstring.
        if "HYDROmod_QAchange" in meta.columns:
            events["HYDROmod_QAchange"] = meta.loc[site_no, "HYDROmod_QAchange"]

    return labels_df, events


def build_database(
    config: PipelineConfig | None = None,
    gage_ids: list[str] | None = None,
    force: bool = False,
) -> None:
    """Build the national BFD database from GAGES-II + NWIS data.

    Writes two Parquet files:
        data/outputs/daily_labels.parquet
        data/outputs/bfd_events.parquet
    and a provenance JSON:
        data/outputs/provenance.json

    Parameters
    ----------
    config : PipelineConfig
        Pipeline parameters; uses defaults if None.
    gage_ids : list[str]
        Override the gage list (e.g. for a test run on a subset).
    force : bool
        If True, re-process even gages with existing cached label files.
    """
    if config is None:
        config = PipelineConfig()

    meta = load_gagesii_metadata()
    if gage_ids is None:
        gage_ids = get_gage_ids(meta)

    rf_model, rf_scaler = load_rf_bfd_model()

    all_labels: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []

    for site_no in tqdm(gage_ids, desc="Building database"):
        labels, events = _process_one_gage(
            site_no, config, meta, rf_model, rf_scaler, force=force
        )
        if labels is not None:
            all_labels.append(labels)
        if events is not None and not events.empty:
            all_events.append(events)

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

    pd.concat(all_labels).to_parquet(DATA_OUTPUTS / "daily_labels.parquet")
    pd.concat(all_events).to_parquet(DATA_OUTPUTS / "bfd_events.parquet")

    prov = create_provenance_record(dataclasses.asdict(config))
    (DATA_OUTPUTS / "provenance.json").write_text(
        json.dumps(prov, indent=2, default=str)
    )
    print(f"Database written to {DATA_OUTPUTS}")
