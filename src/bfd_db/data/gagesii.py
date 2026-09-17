"""
GAGES-II metadata loader.

GAGES-II (Geospatial Attributes of Gages for Evaluating Streamflow) covers
9,322 USGS stream gages with ≥20 complete years of record since 1950 or
active as of WY2009.

Currently sourced from local files rather than the full GAGES-II
'basinchar_and_report_sept_2011.zip':
  data/raw/gagesii/GAGES-II_ref_non_ref.csv  — STAID, CLASS (Ref/Non-ref)
  data/raw/gagesii/site_info.csv             — site_no, dec_lat_va, dec_long_va,
                                                 drain_area_va (sq mi)
  data/raw/gagesii/site_huc.csv              — STAID, HUC2, HUC8; fetched
                                                 directly from the NWIS site
                                                 service (huc_cd field) since
                                                 the official GAGES-II file
                                                 wasn't available — see
                                                 git history for the one-off
                                                 fetch script.

Key columns carried forward to the database:
  STAID            — USGS 8-digit site number (string, zero-padded)
  CLASS            — 'Ref' or 'Non-ref'
  LAT_GAGE         — latitude
  LNG_GAGE         — longitude
  DRAIN_SQKM       — drainage area (km²); converted from NWIS drain_area_va (sq mi)
  HUC2 / HUC8      — hydrologic unit codes, used by analysis/spatial.py and
                      analysis/validation.py for regional grouping / leave-
                      region-out cross-validation

NOT YET AVAILABLE — requires the official GAGES-II basinchar file
(conterm_hydro.txt):
  HYDROmod_QAchange — hydrologic disturbance index, used by database/query.py
"""

import pandas as pd

from bfd_db.config import DATA_RAW

GAGESII_DIR = DATA_RAW / "gagesii"

_SQMI_TO_SQKM = 2.58999


def load_gagesii_metadata() -> pd.DataFrame:
    """Load GAGES-II reference classification + NWIS site attributes.

    Returns a DataFrame indexed by STAID (zero-padded 8-digit string) with
    columns: CLASS, LAT_GAGE, LNG_GAGE, DRAIN_SQKM, HUC2, HUC8.
    """
    classif = pd.read_csv(
        GAGESII_DIR / "GAGES-II_ref_non_ref.csv",
        dtype={"STAID": str},
    )
    classif["STAID"] = classif["STAID"].str.zfill(8)

    site_info = pd.read_csv(
        GAGESII_DIR / "site_info.csv",
        dtype={"site_no": str},
    )
    site_info["STAID"] = site_info["site_no"].str.zfill(8)
    site_info = site_info.rename(
        columns={"dec_lat_va": "LAT_GAGE", "dec_long_va": "LNG_GAGE"}
    )
    site_info["DRAIN_SQKM"] = (
        pd.to_numeric(site_info["drain_area_va"], errors="coerce") * _SQMI_TO_SQKM
    )

    huc = pd.read_csv(GAGESII_DIR / "site_huc.csv", dtype=str)

    meta = classif.merge(
        site_info[["STAID", "LAT_GAGE", "LNG_GAGE", "DRAIN_SQKM"]],
        on="STAID",
        how="left",
    )
    meta = meta.merge(huc, on="STAID", how="left")
    meta = meta.set_index("STAID")
    return meta


def get_gage_ids(meta: pd.DataFrame | None = None) -> list[str]:
    """Return all GAGES-II site IDs as zero-padded 8-digit strings."""
    if meta is None:
        meta = load_gagesii_metadata()
    return meta.index.tolist()


def get_reference_gages(meta: pd.DataFrame | None = None) -> list[str]:
    """Return the 2,057 reference (least-disturbed) gage IDs."""
    if meta is None:
        meta = load_gagesii_metadata()
    return meta[meta["CLASS"] == "Ref"].index.tolist()
