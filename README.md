# Whaling Logs — Data Pipeline

This repository contains code and notebooks for cleaning, infilling, and analyzing digitized 19th-century New England whaling logbooks. The attached document provides and overview of the pipeline that transforms transcribed database entries into a validated scientific dataset.

---
#### Input - csv file from logbooks database (https://logbooks.whoi.edu/)
Website is password protected, you will need to be added as a user to access the database.

#### Output - tiered datasets, data visuals, and summary metadata

Our data pipeline takes in the entries as entered by the archival researches, validates all new entries and makes corrections, and produces four verified versions of the dataset along with dataset visualizations and metadata. Each tier is a superset of the previous, adding entries via coordinate interpolation:

| Tier | Description |
|------|-------------|
| **1** | Original entries only. Duplicates removed, strings standardized, coordinates validated and corrected, wind force mapped to the Beaufort scale. |
| **2** | Tier 1 + entries with coordinates infilled across gaps of ≤ 1 day and ≤ 350 km. |
| **3** | Tier 2 + infilling across gaps of ≤ 2 days and ≤ 400 km. |
| **4** | Tier 3 + infilling across gaps of ≤ 3, 4, and 5 days, all at ≤ 450 km. |

A `TierN_usable` boolean flag marks rows where **all three** of `Latitude_decimal`, `Longitude_decimal`, and `BF Value` are non-null. Visualization notebooks typically filter on this flag (ie when we produce plots of 'Tier N Usable Entries', those plots only include log entries where this value is `True`).

---

## Repository Structure

Each data export lives in its own timestamped directory, where YYYYMMDD is the day the export was started. All directories follow the same layout:

```
YYYYMMDD_export/
├── 01_clean_data.ipynb          # Cleaning exported dataset to produce Tier 1
├── 02_infill_data.ipynb         # Infilling Tier 1 dataset to produce Tiers 2, 3, 4
├── 03_make_plots.ipynb          # Generate and save standard publication figures
├── 04_extra_plots.ipynb         # Some extra plots - optional to run
├── 05_ship_sightings.ipynb      # Regenerate ship sighting figure and stats
├── 06_generate_meta_table.ipynb # Per-logbook metadata & trajectory maps
│
├── csv_files/                   # Raw export CSV + intermediate/final CSVs
├── pkl_files/                   # Pkl DataFrames (Tier1–Tier4)
├── figures/                     # Exploratory figures (04_extra_plots output)
├── manuscript_figures/          # Publication-quality figures (03, 05 output)
├── meta_figs/
│   ├── single_voyages/          # Individual ship trajectory plots
│   └── combined_voyages/        # Multi-ship overlay plots
├── newsletter_figures/          # Public-facing visualizations
├── output_txt_files/            # Processing logs (generated per-export)
├── permanent_txt_files/         # Reference files carried forward across exports
└── utils/
    ├── cleaning.py              # All Tier 1 cleaning functions
    ├── infilling.py             # Tier 2–4 interpolation functions
    ├── s_plots.py               # Standard publication figure functions
    ├── x_plots.py               # Exploratory figure functions
    ├── ship_sightings.py        # Cross-logbook sighting analysis
    ├── meta.py                  # Logbook metadata & trajectory plotting
    └── stats.py                 # Haversine distance & voyage statistics
```

---

## Raw Data Schema

Each row in the raw WordPress export represents one logbook entry. Columns:

| Column | Description |
|--------|-------------|
| `LogBook ID` | Vessel identifier: `"Name (Type) YYYY-YYYY"` (e.g. `"Young Phenix (ship) 1836-1840"`) |
| `ID` | Unique integer ID for this entry |
| `Entry Date` | Date of observation |
| `Latitude` / `Longitude` | Original DMS string (e.g. `"42 15 N"`) |
| `Weather` | Free-text weather description |
| `Wind Direction` | Cardinal or intercardinal direction string |
| `Wind Speed/Force` | Raw wind description (mapped to Beaufort scale during cleaning) |
| `Ship Sightings` | Free-text record of other vessels seen |
| `Page` | Source document page number |
| `Depth` | Water depth measurement |
| `Miscellaneous Observations` | Free-text notes |
| `Cloud Cover` | Categorical cloud cover description |
| `Sea State` | Categorical sea state description |
| `Bottom` | Sea bottom type (if depth sounded) |
| `Landmark` | Geographic reference points mentioned |

The following columns are present in the raw export but dropped at the start of `01_clean_data.ipynb:

| Column(s) | Reason dropped |
|-----------|---------------|
| `Current` | Not used in analysis |
| `Instrumental Observations` | Not used in analysis |
| `2. Ship Heading/Course`, `3. Ship Heading` | Secondary/tertiary heading fields; not consistently populated |
| `2. Wind Direction/Speed/Force/Sea State/Cloud Cover/Weather` | Not consistently populated; dropped without back-filling (see note below) |
| `3. Wind Direction/Speed/Force/Sea State/Cloud Cover/Weather` | Not consistently populated; dropped without back-filling (see note below) |

The `2.` and `3.` wind/weather columns represent additional observations within the same log entry (ie a noon and evening reading). In theory they could be used to recover wind data for entries where the primary observation is missing — Cell 8 of `01_clean_data.ipynb` has commented-out code that checks how often this is actually the case. I looked into it but never ended up implementing the fill, so for now these columns are just dropped. Something to consider for future revisions...

---

## Step 0: Exporting from WordPress

When starting a new export, we select the 'textblocks' format of the dataset in wordpress and then press export. Do this outside of peak work hours. The server the site was built on has other responsibilities and exporting the 130,000 + row (and growing) csv file is laborous. The resulting CSV is automatically named `logentries-export-YYYY-MM-DD.csv` (where the date is the export date). Copy an old directory and replace the old version of the csv with the lateset export in `<new_export_dir>/csv_files/` to run the data processing notebooks on the most recent export. @Alex I also think that the database is an area for potential improvement. 

---

## Step 1: `01_clean_data.ipynb` — Clean dataset to make the Tier 1 dataset

**Input:** `csv_files/logentries-export-YYYY-MM-DD.csv`  
**Output:** `pkl_files/Tier1.pkl`, `csv_files/Tier1.csv`, intermediate CSVs, correction logs in `output_txt_files/`

This step is by far the most work of any. Here we are focused on catching and correcting errors that the logkeepers or archival researchers may have made. We want to correct these not just within the dataset we are creating in Jupyter but also in the database so that we dont have to readdress the same issues during every export.

### 0. Directory Setup

Imports and folder paths are defined relative to the notebook's location and created automatically if missing (`csv_files`, `pkl_files`, `manuscript_figures`, `output_txt_files`).

### 1. Import Data

Near the top of the notebook, adjust the `export_csv` filename to match the one you just downloaded from the database and placed in the `csv_files` directory:

```python
export_csv = 'logentries-export-YYYY-MM-DD.csv'
```

I normally only copy over `permanent_txt_files` and let the rest be auto generated so I am not deleting a bunch of old files.

### 2. Initial Cleanup

#### 2.1 Drop Test Entries & Problem Logbooks

Test entries and logbooks with known systematic errors are dropped at the start. Two lists are maintained:

- **`standard_drops`** — Always dropped (test logbooks, placeholder entries).
- **`temp_drops`** — Dropped until the underlying issues are resolved. I mainly use this for logbooks with known, extensive issues that one of the researchers is working on correcting. 

#### 2.2 DateTime Formatting

Combines the `Entry Date` and `Local Time` columns into a single `Entry Date Time` datetime column. Malformed values that would produce trailing `nan` are set to `NaT`. When saving/reimporting, datetime types are not retained so this step is done redundantly. 

### 3. Remove Duplicate Entries

#### 3.1 Known Duplicate Logbooks

Some ships have multiple overlapping logbooks in the database (e.g. *Leonidas*, *Margaret*). Entries that appear in both are deduplicated by `Entry Date Time`, keeping the more complete record and dropping the overlap.

#### 3.2 Double Dates

`correct_dups(df, dup_ids, log_path)` — **Interactive.** Given a pair of suspected-duplicate IDs, it prints the surrounding logbook context and a side-by-side field comparison, then prompts you to drop one entry, edit a date, or leave both. All decisions are written to `output_txt_files/duplicate_corrections_log.txt`.

Known-good duplicates (entries that appear twice but are legitimately distinct) are tracked in `permanent_txt_files/ok_duplicate_dates.txt` and skipped automatically.

Things we are looking to correct here:
   - **Incorrectly entered dates** — the most common case is a transcriber entering the wrong year for an entry (ie writing 1851 instead of 1852) so we have entries on dates: 01-01-1851, 01-02-1851, 01-03-1851 (...) 01-01-1852, 01-02-1851, 01-03-1852. Here, 01-02-1851 appears 2x with distinct entry IDs so it will be flagged. With these issues we simply correct the year using the `[e]` edit option — you can easily tell which year is right by looking at the surrounding entries and checking whether the sequence of dates is continuous.
   - **Genuinely duplicated entries** — sometimes the same entry was transcribed twice, either accidentally or because a day is split across pages. Just drop one in the dataset.
   - **Two real observations on the same day** — some logbooks have a noon entry and an evening entry recorded separately. These look like duplicates but both should be kept. If the surrounding context and field values look distinct, mark them as a known-good duplicate with `[n]` and add the pair to `ok_duplicate_dates.txt` so they don't get flagged again.

**NOTE:** Legitimate issues like a wrong year entered (and all issues from sections 3 through 6) should be made permanently within the database. I typically just have the db open as I go through my cleaning steps so that I can make corrections as I uncover issues; however, all corrections will be logged so you can run the data cleaning within jupyter and then after the fact make all corrections within the database - I just find making 20 corrections back to back to be more tedious.

### 4. Clean Text Columns

#### 4.1 Page, Depth, and Ship Sightings

| Function | Column(s) Cleaned | Method |
|----------|-------------------|--------|
| `clean_page_column(df)` | `Page` | Exact-match dictionary + NaN list; converts to `float64` |
| `clean_depth_column(df)` | `Depth` | Same approach as `clean_page_column` |

Each function prints any values it does not recognize — add these to the map within the function definition and re-run. When all new strings have been handled you will get the message `'all {depth OR page} values successfully handled.'`

#### 4.2 Common String Cleanup

Sets common non-data strings (e.g. `"none"`, `"not recorded"`) to NaN across `Ship Heading/Course`, `Wind Direction`, and `Wind Speed/Force`. Also normalizes whitespace in free-text columns (`Ship Sightings`, `Miscellaneous Observations`, `Landmark`).

### 5. Clean Wind Data

#### 5.1 Wind Direction

`clean_wind_dirs(df)` — Applies regex patterns to map raw wind direction strings to standard cardinal/intercardinal names. Returns the cleaned DataFrame and a list of values set to NaN.

With each export, inspect the NaN list to confirm we aren't discarding valid directions. Most excluded terms are things like `'BAFFLING/VARIABLE'` or `'N to S'`, but new valid directions may surface and should be added to the map.

`wind_dir_to_numeric(df, col, out_col)` — Converts standardized direction names (e.g. `"NNE"`) to a numeric bearing in degrees, stored in `WD_Bearing`.

#### 5.2 Wind Speed/Force

Converts raw wind descriptions to the **Beaufort scale (0–12)**:

| Function | Purpose |
|----------|---------|
| `init_wind_force_clean(df, col)` | Fix common typos and formatting errors in the raw wind force column before mapping |
| `load_beaufort_map(filename, unique_only)` | Load the Beaufort mapping dictionary from `permanent_txt_files/wind_force_classified.txt` |
| `save_beaufort_map(bf_map, filename)` | Save an updated Beaufort map back to the repo |
| `parse_beaufort_series(df, col, bf_map, ...)` | **Interactive.** Iterates through unmapped wind strings, prompts for BF values, and logs assignments. Mapped values are written to a log file and to `BF Value` in the DataFrame |

The Beaufort mapping is cumulative across exports — `wind_force_classified.txt` grows over time as new strings are encountered. I inherited the initial list and categorized new terms I encountered by matching key words. 

#### 5.3 Sea State, Cloud Cover, and Weather

`clean_remaining_strings(df)` — Applies standardized replacement dictionaries to `Sea State`, `Cloud Cover`, and `Weather` columns.

### 6. Coordinate Cleaning

#### 6.1 Addressing formatting issues and converting from DMS to decimal

Several functions are applied in sequence to catch and correct coordinate entry issues. `normalize_coords` handles simple syntax errors such as missing or extra spaces and uncapitalized cardinal directions. `flag_and_convert_miles` then converts any distance-based entries (recorded in miles) to degree values. Next, we look for a few last issues that sometimes arise. We pass `flag_coords_too_many_digits` as input an to `batch_correct_coords`. If there is an issue, the function will show the error and allow you to enter the corrected version (ie 46 100 N --> 46 10 N). If no issues are detected, the correction wrapper function will return a message stating that no such issues were found. We must now apply a set of hardcoded corrections is applied to entries from the Good Return and Gideon Howland, both of which recorded longitude values exceeding 180°. These are converted to a numerically usable form in postprocessing rather than corrected in the database, as they reflect the original logkeeper's notation. A tradeoff between preserving historical accuracy in the database and producing scientifically usable values downstream. After doing this we can inspect for other entries which exceed expected lat/lon limits and correct any indentified errors using  the `flag_coords_beyond_bounds` and `examine_and_correct_outliers` functions (same wrapper logic as flag too many and batch correct). We run a final cleaning function which handles some edge cases and sets string "nans" to proper `NaN` values and then run `save_invalid_coords` to output txt files containing any invalid entries that slipped through our QC functions. We correct these issues if there are any (easiest method, probably `correct_coord` as it will log the change)

Having made basic formatting/syntax correction we prepare to convert to decimal notations as it's easier for plotting and computing distances. This is done with a few functions

`add_decimal_columns` — calls `dms_to_decimal()` on every row and adds `Latitude_decimal` and `Longitude_decimal` columns where `dms_to_decimal` parses a single `"D M S Dir"` string to a signed decimal degree value.

**Note:** Basic formatting corrections identified and executed should also be made within the database. Corrections like those made for the Good Return will remain in post processing -- I think this is and area where the workflow could be improved (ie move the corrections to a util)

For reference all functions mentioned above, the input objects, and their intended purposes:

| Flag Function | Error Detected |
|---------------|---------------|
| `flag_coords_missing_direction(df, lat_col, lon_col)` | Coordinates missing the N/S or E/W suffix |
| `flag_direction_symbol_errors(df, lat_col, lon_col)` | Swapped N↔S or E↔W |
| `flag_coords_beyond_bounds(df, lat_col, lon_col)` | Latitude > 90° or longitude > 180° |
| `flag_coords_too_many_digits(df, lat_col, lon_col)` | Degree or minute fields with too many digits (likely a transcription error) |

Supporting correction functions:

| Function | Purpose |
|----------|---------|
| `batch_correct_coords(df, flag_func, lat_col, lon_col)` | Run a flag function, then interactively correct each flagged row |
| `correct_coord(df, row_idx, col, log_path, force_both)` | Prompt for a corrected coordinate value for a single row; log the change |
| `apply_coord_corrections(df, corrections, id_col)` | Apply a pre-built dictionary of corrections non-interactively |
| `examine_and_correct_outliers(df, flagged_ids, col, log_path)` | Review and correct outlier rows identified by jump detection |
| `save_invalid_coords(df, lat_col, lon_col, dir_path)` | Export rows with coordinates still invalid after all correction passes |

#### 6.2 Unrealistic Coordinates

`flag_unrealistic_coord_jumps(df, lat_col, lon_col, max_speed_kmh, log_path, window_size)` — Compares consecutive entries within each logbook and flags pairs where the implied speed between observations exceeds `max_speed_kmh`. Logs flagged pairs to `output_txt_files/`.

`inspect_and_correct_logbook_flags(df, logbook_id, ...)` — **Interactive.** Displays flagged entries in context (surrounding rows, a map) and prompts for corrections. Corrections are logged.

`final_coord_cleanup(df, lat_col, lon_col)` — Drops any rows where coordinates are still invalid after all correction passes.

Visualization helpers used during inspection:

| Function | Purpose |
|----------|---------|
| `plot_logbook(df, logbook_id, ...)` | Plot all entries for one logbook on a map, optionally highlighting new or flagged rows |
| `plot_new_entries(df, logbook_id, new_rows, ...)` | Highlight recently added entries in a logbook plot to check for positioning errors |


**Note:** Throughout the notebooks you will see hardcoded changes made directly to the pandas dataframe. I do this mostly so that I dont have to step through the interactive inpection/correction everytime I reload the notebook. ie I inspect and ID corretions that need to be made, hardcode the changes in the notebook (before the flag/inspect steps) so that if I pick up corrections a day later I am not doing work I already did, and then make them in the database (so that I am not making the same changes on the next export).

### 7. Visual Inspection

All new entries (those not present in the previous export) are plotted by logbook to catch any remaining trajectory errors that slipped through the automated flagging. `plot_new_entries()` generates one panel per logbook containing new rows; the scope can be set to plot only the new rows or all rows in each affected logbook.

Normally you catch a few errors here which skip through the logic (this comes from issues like the entry not having entries before or after so a N to S switch is not flagged despite the fact that it is way beyond the vessel path - easy to just spot, harder to code into the functions) and use the correction functions to make those changes 

### 8. Final Standardization & Saving

After all corrections are applied, a final exclusion filter removes any logbooks still flagged as problematic. Then, `map_bf(value)` — Maps multi-category Beaufort values (e.g. `23`, `45`) produced by borderline wind strings to single integer values by taking the lower bound. Valid single-digit values 0–12 are kept as-is. The remapped column replaces `BF Value` and the final Tier 1 outputs (`Tier1.csv` / `Tier1.pkl`) are saved.

---

## Step 2: `02_infill_data.ipynb` — Gapfilling to generate Tiers 2, 3, and 4 datasets

**Input:** `pkl_files/Tier1.pkl`  
**Output:** `pkl_files/Tier{2,3,4}.pkl`, `csv_files/Tier{2,3,4}.csv`

### How Infilling Works

`infill_missing_data(df, columns_to_infill, days_missing, max_distance_km)` fills coordinate gaps using linear interpolation. For each window of consecutive same-logbook rows:

1. The two endpoint rows must both have valid coordinates.
2. All interior rows must be missing the target coordinate(s).
3. The date span must equal exactly `days_missing + 1` days (i.e. a contiguous gap).
4. The great-circle distance between endpoints must be ≤ `max_distance_km`.

If all conditions are met, the interior rows receive linearly interpolated values. The function handles the antimeridian correctly by temporarily shifting longitudes before interpolating.

The notebook calls `infill_missing_data()` three times with progressively relaxed thresholds:

| Pass | Gap Size | Max Distance | Output |
|------|----------|-------------|--------|
| Tier 2 | ≤ 1 day | 350 km | `Tier2.pkl/.csv` |
| Tier 3 | ≤ 2 days | 400 km | `Tier3.pkl/.csv` |
| Tier 4 | ≤ 3, 4, 5 days | 450 km each | `Tier4.pkl/.csv` |

### Infilling Meta Columns Added

For **accepted** gaps (infill applied):

| Column | Description |
|--------|-------------|
| `Infilled` | `True` for all infilled rows |
| `infill_days_missing` | Number of missing days in this gap |
| `infill_type` | `"lat"`, `"lon"`, or `"latlon"` depending on which coordinate(s) were missing |
| `infill_distance_km` | Great-circle span distance between the two endpoints |

For **rejected** gaps (distance or duration exceeded threshold):

| Column | Description |
|--------|-------------|
| `gap_distance_km` | Span distance of the gap that was not infilled |
| `gap_days_missing` | Number of missing days in the rejected gap |
| `gap_type` | Same `lat`/`lon`/`latlon` classification |

### Usability Flag

After all infilling, a boolean `TierN_usable` column is added to each tier:

```python
df['TierN_usable'] = (
    df['Latitude_decimal'].notna() &
    df['Longitude_decimal'].notna() &
    df['BF Value'].notna()
)
```

### Statistical Validation

`calculate_statistical_significance(series1, series2, name1, name2)` — Runs a Mann-Whitney U test and a Kruskal-Wallis test comparing the original and infilled distributions, printing the results. This is run after each tier to confirm that infilling does not significantly shift the wind data distribution.

### Read out

We use `drop_cols_if_present(df, cols)` to remove a few unecessary columns before saving each tiered dataset.

---

## Step 3: `03_make_plots.ipynb` — Standard Publication Figures

**Input:** Any tier PKL (typically Tier 4 as we have shown no statistical difference in the dataset and it gives the most usable data points)  
**Output:** PNG files in `manuscript_figures/`


| Figure | Function | Description |
|--------|----------|-------------|
| Global map | `plot_global(df_coords, df_usable, ...)` | All usable entries as a scatter on a Robinson projection, colored by year with a continuous colorbar |
| Decadal maps | `plot_decadal(df, ...)` | 2×5 subplot grid, one panel per decade (1820s–1890s); viridis colormap across the full year range |
| Seasonal maps | `plot_seasonal(df, ...)` | 2×2 subplot grid (DJF, MAM, JJA, SON); each season has a fixed color (winter blue, spring green, summer orange, autumn pink) |
| Annual distribution | `plot_annual_distribution(df, ...)` | Bar chart of entry count per year |
| Infill boxplots | `plot_infill_boxplot_map(df, ...)` | Side-by-side boxplots of wind force and direction before vs. after infilling, with a map inset showing infilled entry locations |
| Split infill boxplots | `plot_infill_boxplot_map_split(df, ...)` | Same as above but split by an additional grouping variable (e.g. month) |


---

## Step 4: `04_extra_plots.ipynb` — Optional exploratory figures

**Input:** `pkl_files/Tier4.pkl`  
**Output:** PNG files in `figures/`

These figures are generated on demand for inspection and are not part of the standard publication set. I use this notebook to look closer at individual logbooks with extensive issues or make regionally focused figures.

| Figure | Function | Description |
|--------|----------|-------------|
| Single voyage | `plot_single_journey(df, logbook_id, ...)` | Trajectory map for one vessel with markers at the start, end, and southernmost point |
| Regional view | `plot_global(df, ..., region, bounds)` | Same global scatter limited to a named region (see below) |
| Compact decadal | `plot_decadal_compact(df, ...)` | Condensed multi-decade grid for overview comparisons |
| Date-range subset | `plot_time_range(df, ..., start_dt, end_dt)` | Entries filtered to a custom date window |

**Preset region names:** `north_atlantic`, `south_atlantic`, `indian_ocean`, `pacific`, `pacific_ne`, `cape_cod`, `pnw`, `southern_ocean`

Internal helpers used by all plot functions:

| Helper | Purpose |
|--------|---------|
| `_resolve_bounds(region, bounds)` | Convert a region name to a `(W, E, S, N)` bounds tuple and default projection |
| `_apply_extent(ax, bounds)` | Set map extents from a bounds tuple |
| `_add_base_map(ax, draw_labels)` | Add coastlines, land fill, and gridlines to a Cartopy axis |
| `_coerce_to_timestamp(val, role)` | Parse flexible datetime input to a pandas Timestamp |

---

## Step 5: `05_ship_sightings.ipynb` — Cross-Logbook Sighting Pairs

**Input:** `csv_files/Tier1.csv` (and full dataset for target matching)  
**Output:** `manuscript_figures/WindForce_Map_Hists.png`; printed agreement statistics

This notebook identifies pairs of logbook entries where one ship recorded a sighting of another and both logbooks have entries close in time and space. The pipeline:

1. **Build ship catalog** — `build_ship_catalog(df, ship_col)` parses every `LogBook ID` into `(name_key, vessel_type, start_year, end_year)`.

2. **Extract sighting rows** — `filter_ship_sighting_rows(df, ...)` returns only entries with a non-empty `Ship Sightings` field.

3. **Extract vessel names** — `extract_names_from_sighting(text, catalog_names)` uses two pattern strategies: parenthetical vessel names (e.g. `"Name (Ship)"`) and token-level AND-matching against the catalog.

4. **Match to vessels** — `build_sighting_pairs(df, df_sight, ...)` cross-references extracted names against the catalog by name key and year range, producing a `(source_logbook, target_logbook)` pair table.

5. **Materialize close pairs** — `materialize_close_pairs(df, pairs_df, max_km, max_days, prefer_same_day)` finds target logbook rows within `max_days` days and `max_km` km of each sighting. Defaults: ±1 day, ≤ 200 km. Same-day matches are preferred; ties broken by distance.

6. **Compute differences** — `add_pairwise_differences(df_pairs, ...)` adds:
   - `BF_diff`, `BF_abs_diff` — Beaufort scale difference (signed and unsigned)
   - `WD_diff_deg`, `WD_abs_diff_deg` — Circular wind direction difference (−180° to +180°)

7. **Summarize agreement** — `summarize_bf_wd_agreement(df_pairs)` returns the fraction of pairs within ±1 BF and within ±22.5°/45° of wind direction.

8. **Visualize** — `plot_map_with_bf_wd_abs_histograms(df_pairs, ...)` produces a combined figure: a map of sighting locations colored by Beaufort force, plus histograms of ΔBF and ΔWD.

Internal helpers:

| Helper | Purpose |
|--------|---------|
| `_standardize_name(s)` | Strip parentheticals, year ranges, and vessel-type words; lowercase |
| `_parse_logbook_id(log_id)` | Extract `(name_key, vessel_type, start_year, end_year)` from a `LogBook ID` string |
| `_expand_two_digit_year(end_two, start_four)` | Infer the century of a 2-digit year end (e.g. `"52"` to `1852`) |
| `_pick_col(df, candidates)` | Return the first candidate column name that exists in the DataFrame |

---

## Step 6: `06_generate_meta_table.ipynb` — Logbook Metadata

**Input:** `csv_files/Tier{1,2,3,4}.csv`  
**Output:** `csv_files/TierNlogbooks_meta.csv`; trajectory plots in `meta_figs/`

This notebook generates metadata tables as well as overview maps for each vessel. The tables are produced for each tier and include `LogBook ID`, `Total_Entries`, `Total_Usable_Entries`, `Researcher`, and `Repository`. We produce figures of all entries as well as individual plots of each voyage using different colorfills to show temporal progress of the vessel and observed bf values. The Tier 4 meta file, with the `Researcher` column removed, is made available as the overview.tab file in the published dataset version [here](https://dataverse.whoi.edu/dataset.xhtml;jsessionid=729e1dbdb4ba6c25b590303f203c?persistentId=doi%3A10.26027%2FDATAWJBAVU). I use this file to track logbooks that are problem-free and those with ongoing issues. 

**Build metadata df:**

We pull in two datasets: the tiered 1–4 log entries and a logbooks metadata file (pulled from WordPress using the logbooks export, not the log entries export). We use Tier 4. Before we can merge them, we need to reconcile `LogBook ID` strings across the two files. This happens in four passes:

First, `logbook_id_fix_map` resolves truncated IDs — the WordPress UI silently cuts off long logbook names with an ellipsis in the log entries export but not the logbooks export (@Alex someone with more backend experience would just fix this in WordPress, but that is not me — Helen tried and it didn't take). One truncated ID (`"Gage H. Phillips (Schooner) 18…"`) maps to two separate voyages and can't be handled with a simple replacement, so we split it on entry date year instead. Then `standardize_logbook_id(raw)` normalizes spacing, capitalization, and dash formatting, and `date_fix_map` corrects four logbooks where the date range was entered differently across the two source files.

Once the IDs are clean, we group the entries by `LogBook ID` to compute `Total_Entries` and `Total_Usable_Entries`. We then read in the metadata file, apply the same string normalization, deduplicate, standardize repository names via `standard_repo_map`, and left-merge on `LogBook ID` to produce the final `summary_with_meta` df. A validation pass at the end flags any logbooks missing metadata and lists logbooks present in the metadata but absent from the dataset.


**Trajectory visualization:**

We now create the overview and individual voyage visualizations. Our `plot_logbook_with_options(df, logbook_id, ...)` function is capable of creating all required figure versions by adjusting the logbooks included in the `input_df` param or changing the `color_by` option from `"Entry Date"`or `"BF Value"`.

### Sync to drive
I have built a utility script which uploads the finalized metadata, tiered datasets, and figures to Drive. This ensures we keep versions of each dataset somewhere safe. The sync will need to be reconfigured to use new credentials but hat isn't too tricky. The sync scripts use rclone, which you can install from pip or conda. To reconfigure for new credentials, run rclone config and set up a new Google Drive remote named whaling_logbooks_GDrive — it'll walk you through the OAuth flow in your terminal. Once that's done, the sync_data.sh and sync_metadata.sh scripts in utils/ should work as-is.

---

## Output Directory Reference

| Directory | Contents |
|-----------|---------|
| `csv_files/` | Raw export CSV; intermediate cleaning CSVs (`logbook_no_dupes.csv`, `logbooks_clean_strings.csv`, etc.); final Tier 1–4 CSVs; logbook metadata tables |
| `pkl_files/` | Pickled DataFrames for Tier 1–4 (faster to load than CSV for iterative analysis) |
| `figures/` | Exploratory figures from `04_extra_plots.ipynb` |
| `manuscript_figures/` | Publication-ready figures from `03_make_plots.ipynb` and `05_ship_sightings.ipynb` |
| `meta_figs/single_voyages/` | One trajectory plot per logbook from `06_generate_meta_table.ipynb` |
| `meta_figs/combined_voyages/` | Multi-ship overlay trajectory plots |
| `newsletter_figures/` | Public-facing visualizations |
| `output_txt_files/` | Processing logs generated per-export: duplicate corrections, coordinate corrections, Beaufort mapping decisions, coordinate jump flags |
| `permanent_txt_files/` | Reference files carried forward across all exports (see below) |

### Permanent Reference Files

These files are updated incrementally and should be copied from the previous export directory when starting a new one:

| File | Purpose |
|------|---------|
| `ok_duplicate_dates.txt` | Approved known-good duplicates; entries in this list are not flagged by `correct_dups()` |
| `unique_wind_directions.txt` | All unique raw wind direction strings ever encountered; used as a reference when adding new direction mappings |
| `wind_force_classified.txt` | The Beaufort mapping dictionary (`raw string → BF value`); grows as new wind force strings are encountered in `parse_beaufort_series()` |

---

## Correction Logs (`output_txt_files/`)

All interactive correction steps append timestamped entries to text files in `output_txt_files/`. These logs are ephemeral (specific to one export run) and provide a full audit trail:

| Log File | Written By |
|----------|-----------|
| `duplicate_corrections_log.txt` | `correct_dups()` — records each pair reviewed and action taken |
| `coord_corrections_log.txt` | `correct_coord()`, `batch_correct_coords()` — records each coordinate value changed |
| `beaufort_mapping_log.txt` | `parse_beaufort_series()` — records each raw string → BF value assignment |
| `coord_jump_flags_log.txt` | `flag_unrealistic_coord_jumps()` — lists flagged entries with implied speeds |

---

## Automated Bootstrapping

Instead of performing the manual steps below, you can use the `bootstrap.py` script to automatically set up a new export directory. The script will create a new directory named `YYYYMMDD_export` using today's date, copy all required code and reference files from the most recent export (while safely ignoring old outputs like CSVs, PKLs, and logs), and place your newly downloaded CSV in the `csv_files` directory with the correct naming convention.

Additionally, the script will check for a local `conda` installation and offer to automatically download and install Miniforge to your home directory if it is missing. It will also offer to create or update the project's conda environment from the `conda-environment.yml` file.

**Usage:**
```bash
python bootstrap.py /path/to/downloaded/logentries.csv
```

---

## Starting a New Export

When new logbook entries have been added to WordPress and you want to produce an updated dataset:

1. **Duplicate the previous export directory** and rename it `YYYYMMDD_export` (using today's date or the export date).

2. **Delete all generated outputs** inside the new directory, keeping only the permanent reference files and the utils/ folder:
   - Delete: `pkl_files/*.pkl`, `csv_files/*.csv`, `figures/*`, `manuscript_figures/*`, `meta_figs/*`, `output_txt_files/` logs
   - Keep: `permanent_txt_files/`, `utils/`

3. **Export from WordPress** (see Step 0 above). Place the resulting file in `csv_files/` as `logentries-export-YYYY-MM-DD.csv`.

4. **In `01_clean_data.ipynb`**, update the filename near the top of the notebook:
   ```python
   export_csv = 'logentries-export-YYYY-MM-DD.csv'
   ```

5. **Review hardcoded ID corrections** in `01_clean_data.ipynb` (the cells that apply `apply_coord_corrections()` or similar). Sometimes I just hardcode the correction in the notebook for the sake of efficiency. All of these will be log

6. **In `utils/meta.py`**, add any new truncated logbook IDs to `logbook_id_fix_map`. These appear when the WordPress UI cuts off a long logbook name with `"..."`.

7. **Check `permanent_txt_files/wind_force_classified.txt`** for any new wind force strings that will need Beaufort values. New strings will surface during the `parse_beaufort_series()` interactive session in `01_clean_data.ipynb`.

8. **Run notebooks 01 through 06 in order.** Notebooks 01 and 02 have interactive prompts; the rest run to completion without user input.
