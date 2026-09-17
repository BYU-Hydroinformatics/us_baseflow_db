"""
bfd_db — National Baseflow-Dominant Database for the United States.

Pipeline:
  1. data      — download GAGES-II metadata and NWIS daily streamflow
  2. labeling  — classify each day as BFD (1) or non-BFD (0) via ensemble
  3. events    — pool daily labels into BFD events; detect causal onset
  4. database  — assemble, store, and query the two-layer database
  5. analysis  — validation against benchmark and CONUS spatial analysis
"""

__version__ = "0.1.0"
