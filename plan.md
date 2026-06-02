# Overview

The objective of this project is to develop a national baseflow database for the United States, together with the Python tools needed to build, update, and query it.

The database is organized in two layers:

1. **Daily baseflow-dominant (BFD) labels — the foundation.** A comprehensive record of daily streamflow for US stream gages, with each date labeled 0 or 1 to indicate whether the flow on that date is baseflow-dominant. These daily labels are the foundation of both the database and the analysis tools; everything else is derived from them.

2. **Baseflow-dominant (BFD) events — derived from the daily labels.** A table of baseflow-dominant *events* built from the daily 0/1 series — i.e., sequences of consecutive (or near-consecutive) baseflow-dominant days — with columns such as gage ID, event start date, duration, and summary statistics (mean flow, standard deviation, position on the flow-duration curve, etc.). An event-level view is often more directly useful to researchers who want to analyze baseflow-dominant episodes rather than individual days, and it is what lets us identify the *onset* of baseflow-dominant periods for the retrospective PyBFS forecast tests (see [Use Cases](#use-cases)). Because events are computed from the daily series, the daily labels remain the canonical product and the events table is a derived view that can be regenerated with different parameters.

**Scope note — baseflow dominance, not drought.** This database concerns baseflow *dominance* (the composition of streamflow — when flow is sustained primarily by drainage from storage) rather than hydrological *drought* (anomalously low flow relative to the local regime). The two can coincide but are distinct, and we deliberately frame everything around baseflow-dominant days and the sequences they form. This focus is motivated by three questions the database is meant to support: (1) how baseflow has responded to climate change, (2) how baseflow responds to declining groundwater levels, and (3) how well PyBFS can forecast streamflow once a gage enters a baseflow-dominant period (a PyBFS forecast assumes little or no rain over the forecast horizon).

The **primary deliverable is an open-source Python toolchain** that builds the daily labels and derived events from NWIS data and lets users regenerate a clean database on demand — with default or custom parameters — as new gage data become available. We will also distribute periodic frozen, versioned snapshots of the national database for users who simply want the data and for citation in the papers. A dynamic website to explore results (browse, visualize, and optionally generate events for a selected gage) is an optional add-on rather than a maintained core product, to keep long-term overhead low. See [Delivery and Tooling](#delivery-and-tooling).

## Use Cases

There are a number of use cases for this database, including:

- Analyzing trends in baseflow over time and across different regions of the country.
- Analyzing the impact of various factors (e.g., land use, climate change) on baseflow.
- Correlating baseflow changes with groundwater storage changes, including the use of GRACE data.
- Applying the GWBASE algorithm to analyze the correlation between baseflow and groundwater level changes in selected basins.
- Analyzing the predictive skill of PyBFS through a retrospective analysis of the database to identify sequences of baseflow-dominant days where forecast skill can be evaluated.

## Background: Approaches to Identifying Baseflow

Historically, researchers have used a variety of methods to identify baseflow-dominant days, including standard baseflow separation techniques. Aghababaei et al. (2025) developed a machine learning-based approach to identify baseflow-dominant days, which has shown promise in improving the accuracy of baseflow identification. Xie et al. (2020) evaluated several typical methods for baseflow separation in the contiguous United States, providing insights into the strengths and weaknesses of each method. They used the Strict algorithm to judge the performance of each method, which is a commonly used algorithm for baseflow separation.

Each of these techniques has its own advantages and disadvantages, and the choice of method will depend on the specific research question being addressed. For example, machine learning-based approaches perform well when compared with the hand-labeled datasets they were trained on, but may not perform as well when applied to new data that differs from the training data. The separation algorithms can also produce a baseflow estimate that is greater than the total streamflow. On the other hand, standard baseflow separation techniques can perform well on some gages but not on others, and may require manual tuning of parameters. In our experience, the Strict algorithm is too restrictive in some cases, and in other cases includes data points high on the recession limb of the hydrograph that are not baseflow-dominant.

Our hypothesis is that an ensemble approach that combines multiple methods for identifying baseflow-dominant days will outperform any single method. By leveraging the strengths of each method, we can improve the accuracy and robustness of our baseflow identification. We will evaluate the performance of our ensemble approach using a combination of hand-labeled datasets and standard baseflow separation techniques, including the Strict algorithm. For the separation techniques, daily streamflow values will be considered baseflow-dominant if the baseflow component is within X% of the total streamflow. The final determination as to whether a day is baseflow-dominant will be made by the ensemble approach — for example, if a day is labeled as baseflow-dominant by a majority (or some percentage) of the methods. Users of the database and associated tools will be able to adjust these thresholds as needed for their specific applications.

# Open Research Questions and Tasks

The following questions must be resolved through literature review and analysis before and during database development. They are grouped by topic.

## 1. BFD Day Labeling

How do we label a day as baseflow-dominant? Options to evaluate include strict baseflow, baseflow within X% of total flow, the current ML-based approach, and an ensemble approach that combines multiple methods. We need to evaluate all of these methods (see [Methods](#methods)) and determine how to combine them.

## 2. BFD Event Definition

A BFD *event* is a sequence of baseflow-dominant days derived from the daily 0/1 labels. Because the daily labels are the foundation, the two difficulties originally raised — a brief **excursion** (flow jumps above the threshold for a day or two) and a longer **interruption** (a short non-BFD stretch inside an otherwise baseflow-dominant period) — are the same phenomenon at the event level: a short run of non-BFD days inside a longer run of BFD days. We therefore need only one mechanism to handle both.

**Recommended definition: inter-event time pooling.** Following the streamflow-drought event literature (Tallaksen et al., 1997; Fleig et al., 2006) — whose event-pooling machinery is method-agnostic even though we are not studying drought — adapted to a binary daily series:

1. Identify candidate events as maximal runs of BFD days (label = 1).
2. Pool two adjacent runs into a single event if the non-BFD gap between them is ≤ `t_c` days (the inter-event time criterion). This single parameter handles both excursions and interruptions.
3. Discard pooled events shorter than a minimum duration `d_min` days.

Worked example: in a 10-day stretch with 8 BFD days and a 2-day interior gap, the result is a single event if `t_c ≥ 2`, and two separate events otherwise.

The tunable parameters — the daily-label threshold `X%`, the gap tolerance `t_c`, and the minimum duration `d_min` — are exposed to users and are the subject of the parameter-sensitivity analysis (see [Event Generation](#5-event-generation) and the BFD Website paper). For the national database we will select standard default values, informed by literature review and by analysis against the expert-labeled dataset (see [Performance Evaluation](#6-performance-evaluation)).

**Onset detection.** Cataloging an event uses the full record, so the event start is known only in hindsight (pooling and `d_min` look ahead). The retrospective PyBFS forecast-skill study (second paper; see [PyBFS Algorithm](#pybfs-algorithm)) additionally requires a *causal* notion of onset — declaring that a gage has entered a baseflow-dominant period using only data available up to that day (e.g., the first day of a confirmed run of `k` consecutive BFD days, with `k` tunable). We will define this causal onset detector alongside the cataloging definition.

## 3. Database Design

How do we store the database, and what statistics should we compute for each event? At a minimum the events table will include gage ID, event start date, and duration; candidate statistics include mean flow, standard deviation, and where the event falls on the flow-duration curve. We also need to decide how to store the underlying daily 0/1 labels from which events are derived. Each gage should additionally carry its GAGES-II metadata — including the reference/non-reference classification and a disturbance/regulation attribute — so that both the daily labels and the events can be filtered by degree of disturbance (see [Database Scope](#4-database-scope)).

## 4. Database Scope

The stored database will be built from the full GAGES-II gage set (9,322 gages), using daily discharge from NWIS, while retaining the ability to generate events on demand for any other gage selected on the website (these on-demand events would not be persisted in the database). Because baseflow separation assumes natural catchment drainage, regulation matters, and our approach is to keep national coverage while making the degree of disturbance explicit:

- Carry the GAGES-II **reference/non-reference classification** and a continuous **disturbance/regulation attribute** (e.g., dam storage and the GAGES-II hydrologic-disturbance index) as first-class metadata on every gage, propagated to its events, so users can filter to relatively natural basins.
- **Validate primarily on reference gages** (2,057 sites), where labels are physically meaningful and checkable (see [Performance Evaluation](#6-performance-evaluation)).
- **Phase the build:** populate and validate the reference gages first, then extend to the non-reference gages with the regulation flags in place.

**Regulation caveat.** On heavily regulated gages, a BFD label describes the composition of the *observed* flow — which may include reservoir releases, diversions, or return flows — not the catchment's natural drainage. Such labels are still recorded, but they are less interpretable for the climate- and groundwater-related analyses, and the regulation metadata lets those studies subset to least-disturbed basins.

A remaining open sub-question is how to handle record gaps, short records, and zero-flow/intermittent or snow-affected regimes.

## 5. Event Generation

How do we generate the events for the database?

- Determine standard parameters for the database (e.g., the X% threshold and other parameters).
- Publish code so that users can generate their own database with different parameters.
- Optionally expose event generation for a specific gage with user-selected parameters through a dynamic website (see [Delivery and Tooling](#delivery-and-tooling)); the same capability is always available directly through the toolchain.

## 6. Performance Evaluation

This concerns the first paper — validating the daily labels and the derived event database. Evaluation operates at two levels:

- **Daily classification.** Compare the ensemble (and the individual methods) against an independent, expert-labeled reference set — e.g., the expert-labeled BFD periods of Aghababaei et al. (2025), which cover 182 USGS gages — using precision, recall, and F1. Hold the reference labels out of any ML training, and use spatial (leave-region-out) cross-validation so that reported performance reflects transfer to unseen basins rather than memorization.
- **Event level.** Evaluate how well the derived events match reference events under the chosen pooling parameters: event hit/miss rates, onset-date error, and duration error.

We will also benchmark against established methods (e.g., the Strict algorithm) as in Xie et al. (2020). Because there is no physical ground truth for "baseflow-dominant," validation against hand labels is necessarily partly circular when those labels also train the ML method; the held-out, spatially cross-validated design above is intended to mitigate this. The separate forecast-skill evaluation (PyBFS, second paper) is described with the PyBFS Forecast Skill paper below.

# Data Sources

Two complementary USGS sources define the data for this project:

- **Gage selection — GAGES-II.** We recommend the USGS GAGES-II dataset (Geospatial Attributes of Gages for Evaluating Streamflow) as the basis for the set of stream gages in the national database. GAGES-II is a geospatial/attribute dataset that links USGS stream gages to the characteristics of their upstream watersheds; it is *not* primarily a streamflow time series. It includes 9,322 USGS stream gages whose watersheds are in the U.S. (including Alaska, Hawaii, and Puerto Rico). Sites were included if they had either 20+ complete years of discharge record since 1950 or were active as of water year 2009. A key feature is the reference vs. non-reference classification: USGS classifies 2,057 of the 9,322 sites as reference gages — those representing relatively least-disturbed hydrologic conditions within broad ecoregions — and the remaining 7,265 as non-reference. This classification is valuable for our work because it lets us distinguish baseflow behavior in relatively natural basins from basins with substantial human alteration, and it supports regionalization, watershed comparison, and climate/land-use analyses.

- **Daily streamflow values — NWIS.** GAGES-II provides the gage/catchment attributes and spatial files, but not the daily discharge record. The actual daily streamflow values for the selected gages will be downloaded from the USGS National Water Information System (NWIS), which provides daily streamflow data for thousands of gages across the United States. We will aggregate these data into daily values for each gage.

Thus the stored national database is built from the GAGES-II gage set with daily discharge retrieved from NWIS (see [Database Scope](#4-database-scope)), while retaining the ability to generate events on demand for any other NWIS gage not in GAGES-II.

# Methods

We will use three main sets of methods to identify baseflow-dominant days: (1) machine learning-based approaches, (2) standard baseflow separation techniques, and (3) the PyBFS algorithm. The daily labels produced by these methods (combined via the ensemble approach) form the foundation from which BFD/drought events are derived.

## Machine Learning-Based Approach

This will be based on the Aghababaei et al. (2025) approach, which uses a machine learning model to identify baseflow-dominant days based on a variety of features derived from the streamflow data. We will run this approach on our dataset to get a classification for each day.

## Standard Baseflow Separation Techniques

For baseflow separation, we will use the algorithms provided in the `baseflowx` Python package.

- https://github.com/BYU-Hydroinformatics/baseflowx
- https://pypi.org/project/baseflowx/
- https://baseflow-explorer.onrender.com/
- https://baseflowx.readthedocs.io/en/latest/

We will need to explore the parameters used by the baseflow separation algorithms to determine the optimal settings for our application. When regenerating the national database, we will use the same parameters for all gages to ensure consistency. However, users of the database and associated tools will be able to adjust these parameters as needed for their specific applications.

## PyBFS Algorithm

PyBFS is a Python implementation of the USGS Baseflow Separation (BFS) model. Unlike the digital-filter techniques in `baseflowx`, PyBFS uses a non-linear state-space reservoir framework in which baseflow is inferred from an (unmeasured) upstream storage state, partitioning total streamflow into baseflow, surface flow, and direct runoff. PyBFS serves **two roles** in this project:

1. **Labeling.** As one of the baseflow separation methods feeding the ensemble (see below), we use its baseflow estimate to classify whether a day is baseflow-dominant.
2. **Forecasting.** PyBFS can also project baseflow forward assuming zero precipitation over the forecast horizon — essentially a recession forecast driven by the drainage of existing storage. This is the capability we exercise in the retrospective forecast test: once a gage enters a baseflow-dominant period (identified from the daily labels/events), we use PyBFS to forecast streamflow and evaluate its skill (see [Use Cases](#use-cases) and the PyBFS Forecast Skill paper).

- https://pybfs.readthedocs.io/en/latest/
- https://pypi.org/project/pybfs/

## Ensemble Approach

For the ensemble approach, we will combine the results from the machine learning-based approach, the standard baseflow separation techniques, and the PyBFS algorithm. These three span distinct method families — a trained classifier, recursive digital filters, and a storage-based state-space model — so they are at least partially independent, which is what makes combining their votes worthwhile rather than simply reinforcing a shared bias. For the standard baseflow separation techniques and the PyBFS algorithm, we will consider a day to be baseflow-dominant if the baseflow component is within X% of the total streamflow.

The final determination as to whether a day is baseflow-dominant will be made by the ensemble approach — for example, if a day is labeled as baseflow-dominant by a majority (or some percentage) of the methods. We will evaluate different methods for combining these results, such as majority voting or weighted averaging, to determine the best approach for identifying baseflow-dominant days.

## Event Derivation

BFD events are derived from the daily 0/1 labels by identifying runs of baseflow-dominant days, subject to rules for tolerating short excursions and interruptions (see [BFD Event Definition](#2-bfd-event-definition)). For each event we will compute and store summary statistics (e.g., duration, mean flow, standard deviation, position on the flow-duration curve). Because events are a derived view of the daily labels, they can be regenerated with different parameters without recomputing the underlying daily classification.

# Delivery and Tooling

To keep long-term maintenance overhead low, the project is **tools-first**:

- **Open-source toolchain (primary deliverable).** A parameterized, reproducible Python pipeline that builds the daily BFD labels and derived events from NWIS data. Users regenerate a clean database whenever they want, with default or custom parameters. This pipeline is also the methods contribution of the first paper.
- **Versioned snapshots.** We will periodically publish frozen, versioned snapshots of the national database (approved data only) to a citable data repository (e.g., HydroShare, Zenodo, or USGS ScienceBase), on our own schedule (e.g., per paper or annually). These give non-coding users the data directly and provide stable, citable versions for the papers.
- **Optional dynamic website.** A dynamic site to explore results — browse and visualize events, and optionally generate events for a selected gage with user-chosen parameters — may be added later. It is treated as an optional add-on, not a maintained core product.

**Data currency, provisional data, and reproducibility.** NWIS distinguishes *provisional* daily values (recent, subject to revision) from *approved* (finalized) values. Because delivery is tools-first, currency is largely the user's choice at regeneration time, but we adopt two conventions so results are reproducible and comparable:

- Store the NWIS approval/qualification status with the daily data and propagate a "contains provisional data" flag to any event that touches provisional days, so provisional results can be filtered. Frozen snapshots are built from approved data only.
- Stamp every generated database — whether a user regeneration or a published snapshot — with a provenance record: NWIS retrieval date, method/package versions (ML model, baseflowx, PyBFS), and the parameter set (`X%`, `t_c`, `d_min`, `k`). This makes any label or event reproducible and explains differences between runs.

# Potential Papers

The two primary, sequential deliverables for this project are the database/tools paper (#1) and the PyBFS forecast-skill paper (#2). The database is also intended to support follow-on work by another student (Xueyi Li), listed separately below.

1. **BFD Database** (first paper) — description of the database and tools, methods, and literature review. Includes the labeling ensemble, the event definition, and the validation in [Performance Evaluation](#6-performance-evaluation). It also covers the open-source toolchain (including the ability to label BFD days and define events with user-selected parameters), a parameter sensitivity analysis (how results change with different parameters for defining BFD days and events), and — optionally — a dynamic website to explore results. As a sample application demonstrating the database, the paper will include a spatial analysis identifying spatial patterns in BFD events across the country (similar to Ryan's WRR paper).
2. **PyBFS Forecast Skill** (second paper) — use the database to evaluate the skill of PyBFS forecasts. A PyBFS forecast simulates the expected baseflow recession over the next `N` days assuming no precipitation, so it is most valuable on a declining limb and is intended for operational use as a conservative baseline within a larger streamflow-forecasting system. Evaluation design:
   - **Idealized (recession) skill.** On historical baseflow-dominant periods with low/no basin-averaged precipitation over the forecast horizon (precip from a gridded product such as PRISM, Daymet, or the CAMELS forcings, with an explicit low-precip threshold), compare PyBFS forecasts to observed flow. This isolates the recession physics from the no-rain assumption and yields a best-case, conditional skill.
   - **Conservative-baseline check.** On periods where precipitation *did* occur, test whether observed flow stays at or above the PyBFS forecast (i.e., the baseline rarely over-predicts) — the property that makes it useful as a floor. Rain periods are therefore kept, not discarded.
   - **Operational (causal) skill.** Fire the forecast at a causal onset (no look-ahead; see [BFD Event Definition](#2-bfd-event-definition)) without knowing whether rain will follow, for an honest operational skill number.
   - **Reference forecasts.** Establish skill relative to trivial baselines — persistence and a one-parameter analytic exponential recession — then compare against NWM and ML methods such as LSTM, positioning PyBFS as the conservative baseline those should beat.
   - **Stratification and scope.** Report skill as a function of lead time `N` and position on the recession limb; restrict the idealized test to reference (unregulated) gages; and stratify by season to expose evapotranspiration and snowmelt confounders.
## Follow-on work supported by the database (Xueyi Li)

These studies are planned by another student (Xueyi Li); this project's database and tools are intended to support them.

- **BFD Events and Groundwater Storage Changes** — analyze the correlation between BFD events and groundwater storage changes, including the use of GRACE data, and apply the GWBASE algorithm to analyze the correlation between baseflow and groundwater level changes in selected basins.
- **BFD Events and Climate Change** — analyze the impact of climate change on BFD events, including changes in their frequency, duration, and intensity over time.

# References

Aghababaei, A., et al. (2025). Development and Comparison of Methods for Identification of Baseflow-Dominant Periods in Streamflow Records. *Water*, 17(21), 3083. https://doi.org/10.3390/w17213083

Fleig, A. K., Tallaksen, L. M., Hisdal, H., & Demuth, S. (2006). A global evaluation of streamflow drought characteristics. *Hydrology and Earth System Sciences*, 10(4), 535–552. https://doi.org/10.5194/hess-10-535-2006

Tallaksen, L. M., Madsen, H., & Clausen, B. (1997). On the definition and modelling of streamflow drought duration and deficit volume. *Hydrological Sciences Journal*, 42(1), 15–33. https://doi.org/10.1080/02626669709492003

Xie, J., Liu, X., Wang, K., Yang, T., Liang, K., & Liu, C. (2020). Evaluation of typical methods for baseflow separation in the contiguous United States. *Journal of Hydrology*, 583, 124628. https://doi.org/10.1016/j.jhydrol.2020.124628
