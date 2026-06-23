# ship_pairing.py
import re
import math
from typing import Optional, Tuple, Set, Iterable, List
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# Column constants (exported)
ID_COL     = 'ID'
SHIP_COL   = 'LogBook ID'
DATE_COL   = 'Entry Date Time'
LAT_COL    = 'Latitude_decimal'
LON_COL    = 'Longitude_decimal'
WDBEAR_COL = 'WD_Bearing'
BF_COL     = 'BF Value'
SIGHT_COL  = 'Ship Sightings'

# Common vessel type words
VESSEL_TYPES = [
    "bark","barque","ship","schooner","brig","brigantine","steamer","steamship",
    "sloop","ketch","cutter","whaleship","bgt","bgt.","bk","bk.","sch","sch."
]

# Text / parsing helpers
def _standardize_name(s: str) -> str:
    """
    Canonicalize a vessel name for matching:
      - remove parentheticals: "(Ship)", "(Barque)", "(Hope)" ...
      - drop year ranges & standalone years
      - drop type words (bark/ship/schooner/etc.)
      - keep letters/digits/space/hyphen/apostrophe; lowercase
    """
    if not isinstance(s, str): return ""
    s = s.replace("’", "'")
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"\([^)]*\)", " ", s)                     # remove parentheses blocks
    s = re.sub(r"\b\d{3,4}\s*-\s*\d{2,4}\b", " ", s)     # remove year ranges
    s = re.sub(r"\b\d{3,4}\b", " ", s)                   # remove lone years
    types_pat = r"(?:^|\s)(?:" + "|".join(map(re.escape, VESSEL_TYPES)) + r")(?:\s|$)"
    s = re.sub(types_pat, " ", s, flags=re.IGNORECASE)   # drop type words
    s = re.sub(r"[^a-zA-Z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _expand_two_digit_year(end_two: int, start_four: int) -> int:
    """Expand 2-digit end year like '52' using century from start year."""
    century = start_four // 100
    candidate = century * 100 + end_two
    if candidate < start_four:
        candidate += 100
    return candidate

def _parse_logbook_id(log_id: str) -> Tuple[str, Optional[str], Optional[int], Optional[int]]:
    """
    'Young Phenix (ship) 1836-1840' -> (name_key, vessel_type, start_year, end_year)
    vessel_type normalized if known; years may be None if not parseable.
    """
    if not isinstance(log_id, str) or not log_id.strip():
        return "", None, None, None

    s = log_id.replace("’", "'")
    s = re.sub(r"[–—]", "-", s)
    m = re.search(r"^(.*?)\s*(?:\(([^)]+)\))?\s*(\d{4})\s*-\s*(\d{2,4})\s*$", s.strip())
    if m:
        raw_name = m.group(1).strip()
        vessel_type = (m.group(2) or "").strip().lower() or None
        y1 = int(m.group(3))
        y2_raw = m.group(4)
        y2 = int(y2_raw) if len(y2_raw) == 4 else _expand_two_digit_year(int(y2_raw), y1)
    else:
        raw_name, vessel_type, y1, y2 = s.strip(), None, None, None

    name_key = _standardize_name(raw_name)
    if vessel_type and _standardize_name(vessel_type) not in VESSEL_TYPES:
        vessel_type = None
    return name_key, vessel_type, y1, y2

# Catalog and filtering
def build_ship_catalog(df_all: pd.DataFrame, ship_col: str = SHIP_COL) -> pd.DataFrame:
    """
    Build a catalog of unique vessels parsed from LogBook IDs.
    Columns: ship_col, name_key, vessel_type, start_year, end_year
    """
    cat = (
        df_all.loc[df_all[ship_col].notna(), [ship_col]]
              .drop_duplicates()
              .assign(
                  name_key    = lambda d: d[ship_col].apply(lambda x: _parse_logbook_id(x)[0]),
                  vessel_type = lambda d: d[ship_col].apply(lambda x: _parse_logbook_id(x)[1]),
                  start_year  = lambda d: d[ship_col].apply(lambda x: _parse_logbook_id(x)[2]),
                  end_year    = lambda d: d[ship_col].apply(lambda x: _parse_logbook_id(x)[3]),
              )
    )
    return cat[cat['name_key'] != ""].copy()

def filter_ship_sighting_rows(
    df: pd.DataFrame,
    id_col: str = ID_COL, ship_col: str = SHIP_COL, date_col: str = DATE_COL,
    lat_col: str = LAT_COL, lon_col: str = LON_COL, wdb_col: str = WDBEAR_COL,
    bf_col: str = BF_COL, sight_col: str = SIGHT_COL
) -> pd.DataFrame:
    """
    Return only rows where a ship sighting is mentioned, with slim set of columns.
    """
    need = [id_col, ship_col, date_col, lat_col, lon_col, wdb_col, bf_col, sight_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    out = df.loc[df[sight_col].notna() & (df[sight_col].astype(str).str.strip() != ''), need].copy()
    out[date_col] = pd.to_datetime(out[date_col], errors='coerce')
    out[lat_col]  = pd.to_numeric(out[lat_col], errors='coerce')
    out[lon_col]  = pd.to_numeric(out[lon_col], errors='coerce')
    out[bf_col]   = pd.to_numeric(out[bf_col], errors='coerce')
    return out


# Sighting name extraction
def extract_names_from_sighting(text: str, catalog_names: Set[str]) -> Set[str]:
    """
    Return set of candidate vessel name_keys found in a sighting text.
    - Prefer patterns like 'bark (hope)' -> 'hope'
    - Also include catalog names whose tokens all appear in the text
    """
    if not isinstance(text, str) or not text.strip():
        return set()

    t = text.replace("’", "'")
    t = re.sub(r"[–—]", "-", t).lower()

    names: Set[str] = set()

    # 1) Extract "type (Name)" -> Name
    type_pat = r"\b(?:" + "|".join(VESSEL_TYPES) + r")\s*\(([^)]+)\)"
    for inner in re.findall(type_pat, t, flags=re.IGNORECASE):
        key = _standardize_name(inner)
        if key:
            names.add(key)

    # 2) Tokenized AND-match against catalog names
    tokens = set(re.findall(r"[a-z]+(?:'[a-z]+)?", t))
    for name_key in catalog_names:
        name_tokens = set(name_key.split())
        if name_tokens and name_tokens.issubset(tokens):
            names.add(name_key)

    return names


# Pairing by name & year overlap
def build_sighting_pairs(
    df_all: pd.DataFrame,
    df_sight: pd.DataFrame,
    id_col: str = ID_COL, ship_col: str = SHIP_COL, date_col: str = DATE_COL,
    lat_col: str = LAT_COL, lon_col: str = LON_COL, wdb_col: str = WDBEAR_COL,
    bf_col: str = BF_COL, sight_col: str = SIGHT_COL
) -> pd.DataFrame:
    """
    For each sighting row, match target vessels by name (from catalog)
    AND ensure the sighting year falls within [start_year, end_year].
    Creates Target_ID as '<Printable Vessel Name>__<YYYY>' using sighting year.
    """
    catalog = build_ship_catalog(df_all, ship_col=ship_col)
    catalog_names = set(catalog['name_key'].unique())

    rows: List[dict] = []
    for _, row in df_sight.iterrows():
        src_name_key = _standardize_name(row[ship_col])  # to avoid pairing to self
        sight_dt = pd.to_datetime(row[date_col], errors='coerce')
        if pd.isna(sight_dt):
            continue
        sight_year = int(sight_dt.year)

        cand_names = extract_names_from_sighting(str(row[sight_col]), catalog_names)
        cand_names.discard(src_name_key)

        for name_key in cand_names:
            hits = catalog.loc[
                (catalog['name_key'] == name_key) &
                catalog['start_year'].notna() & catalog['end_year'].notna() &
                (catalog['start_year'] <= sight_year) & (catalog['end_year'] >= sight_year)
            ]
            if hits.empty:
                continue

            for _, hit in hits.iterrows():
                printable_name = hit[ship_col].split('(')[0].strip()
                target_id = f"{printable_name}__{sight_year}"
                rows.append({
                    'ID_1': row[id_col],
                    'LogBookID_1': row[ship_col],
                    'Date_1': sight_dt,
                    'Latitude_1': row[lat_col],
                    'Longitude_1': row[lon_col],
                    'Wind Force_1': row[bf_col],
                    'Bearing_1': row[wdb_col],
                    'Sight_Text': row[sight_col],

                    'Target_LogBookID': hit[ship_col],
                    'Target_NameKey': name_key,
                    'Target_Start_Year': int(hit['start_year']),
                    'Target_End_Year': int(hit['end_year']),
                    'Target_ID': target_id,
                })

    return pd.DataFrame(rows)

# Materialize close-in-time/space matches
def materialize_close_pairs(
    df_all: pd.DataFrame,
    pairs_df: pd.DataFrame,
    max_km: float = 200.0,
    max_days: int = 1,
    ship_col: str = SHIP_COL, date_col: str = DATE_COL,
    lat_col: str = LAT_COL, lon_col: str = LON_COL, bf_col: str = BF_COL, wdb_col: str = WDBEAR_COL,
    prefer_same_day: bool = True
) -> pd.DataFrame:
    """
    For each proposed (source sighting → target logbook) row, find target rows within +/- max_days,
    compute distances, and return AT MOST ONE match per sighting/target pair.

    Preference:
      1) If any same-day matches exist (Delta_days == 0), keep the one with smallest Distance_km,
         breaking ties by earliest Date_2.
      2) Otherwise, consider +/- max_days; choose row with smallest Delta_days, then smallest
         Distance_km, then earliest Date_2.

    Output columns: *_1 (source sighting) + *_2 (matched target row) + Distance_km, Delta_days.
    """
    tracks = df_all.copy()
    tracks[date_col] = pd.to_datetime(tracks[date_col], errors='coerce')
    tracks[lat_col]  = pd.to_numeric(tracks[lat_col], errors='coerce')
    tracks[lon_col]  = pd.to_numeric(tracks[lon_col], errors='coerce')

    out_rows = []
    time_window = pd.Timedelta(days=max_days)

    for _, pr in pairs_df.iterrows():
        tgt_log = pr['Target_LogBookID']
        dt1 = pd.to_datetime(pr['Date_1'], errors='coerce')

        # Basic guards
        if pd.isna(dt1) or not isinstance(tgt_log, str):
            continue
        if pd.isna(pr['Latitude_1']) or pd.isna(pr['Longitude_1']):
            continue

        # Candidate rows for that target logbook near the sighting date
        cand = tracks.loc[
            (tracks[ship_col] == tgt_log) &
            tracks[date_col].notna() &
            tracks[lat_col].notna() & tracks[lon_col].notna(),
            [ID_COL, ship_col, date_col, lat_col, lon_col, bf_col, wdb_col]
        ]
        if cand.empty:
            continue

        cwin = cand.loc[(cand[date_col] >= dt1 - time_window) & (cand[date_col] <= dt1 + time_window)].copy()
        if cwin.empty:
            continue

        # Compute distance & day difference
        def _row_dist(r):
            try:
                return geodesic((float(pr['Latitude_1']), float(pr['Longitude_1'])),
                                (float(r[lat_col]),      float(r[lon_col]))).kilometers
            except Exception:
                return np.nan

        cwin['Distance_km'] = cwin.apply(_row_dist, axis=1)
        cwin = cwin.loc[cwin['Distance_km'].notna() & (cwin['Distance_km'] <= max_km)].copy()
        if cwin.empty:
            continue

        # Integer day difference (absolute)
        cwin['Delta_days'] = (cwin[date_col] - dt1).dt.days.abs()

        # Selection logic: prefer same-day; else nearest day then nearest distance
        chosen = None
        if prefer_same_day:
            same = cwin.loc[cwin['Delta_days'] == 0]
            if not same.empty:
                chosen = same.sort_values(['Distance_km', date_col]).head(1)
        if chosen is None or chosen.empty:
            chosen = cwin.sort_values(['Delta_days', 'Distance_km', date_col]).head(1)

        if chosen.empty:
            continue

        r = chosen.iloc[0]
        out_rows.append({
            # Source (ship 1)
            'ID_1': pr['ID_1'],
            'LogBookID_1': pr['LogBookID_1'],
            'Date_1': pr['Date_1'],
            'Latitude_1': pr['Latitude_1'],
            'Longitude_1': pr['Longitude_1'],
            'Wind Force_1': pr['Wind Force_1'],
            'Bearing_1': pr['Bearing_1'],

            # Target (ship 2)
            'ID_2': r[ID_COL],
            'LogBookID_2': r[ship_col],
            'Date_2': r[date_col],
            'Latitude_2': r[lat_col],
            'Longitude_2': r[lon_col],
            'Wind Force_2': r[bf_col],
            'Bearing_2': r[wdb_col],

            # Meta
            'Sight_Text': pr['Sight_Text'],
            'Target_ID': pr.get('Target_ID', None),
            'Distance_km': float(r['Distance_km']),
            'Delta_days': int(r['Delta_days']),
        })

    return pd.DataFrame(out_rows)


def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns found: {candidates}")

def add_pairwise_differences(
    df_pairs: pd.DataFrame,
    bf1_col: str = None, bf2_col: str = None,
    wd1_col: str = None, wd2_col: str = None
) -> pd.DataFrame:
    """
    Adds signed and absolute differences with strict NaN rules:
      - If either BF value is NaN -> BF_diff and BF_abs_diff are NaN.
      - If either bearing is NaN -> WD_diff_deg and WD_abs_diff_deg are NaN.

    Creates:
      BF_diff, BF_abs_diff, WD_diff_deg (signed in [-180,180]), WD_abs_diff_deg (0..180).
    """
    out = df_pairs.copy()

    # Auto-detect column names if not specified
    bf1_col = bf1_col or _pick_col(out, ['BF Value_1', 'Wind Force_1', 'BF_1', 'Beaufort_1'])
    bf2_col = bf2_col or _pick_col(out, ['BF Value_2', 'Wind Force_2', 'BF_2', 'Beaufort_2'])
    wd1_col = wd1_col or _pick_col(out, ['Bearing_1', 'WD_Bearing_1', 'Wind Direction_1'])
    wd2_col = wd2_col or _pick_col(out, ['Bearing_2', 'WD_Bearing_2', 'Wind Direction_2'])

    # Coerce to numeric
    bf1 = pd.to_numeric(out[bf1_col], errors='coerce')
    bf2 = pd.to_numeric(out[bf2_col], errors='coerce')
    wd1 = pd.to_numeric(out[wd1_col], errors='coerce')
    wd2 = pd.to_numeric(out[wd2_col], errors='coerce')

    # Validity masks
    bf_valid = bf1.notna() & bf2.notna()
    wd_valid = wd1.notna() & wd2.notna()

    # BF differences (NaN if any input NaN)
    bf_diff = bf1 - bf2
    out['BF_diff'] = np.where(bf_valid, bf_diff, np.nan)
    out['BF_abs_diff'] = np.where(bf_valid, np.abs(bf_diff), np.nan)

    # Circular wind-direction diff (signed in [-180,180]); NaN if any input NaN
    wd_raw = (wd1 - wd2 + 180.0) % 360.0 - 180.0
    out['WD_diff_deg'] = np.where(wd_valid, wd_raw, np.nan)
    out['WD_abs_diff_deg'] = np.where(wd_valid, np.abs(wd_raw), np.nan)

    return out


def plot_map_with_bf_wd_abs_histograms(df_pairs: pd.DataFrame, figures_path=None, save=False, filename='WindForce_Map_Hists.png'):
    """
    Combined figure:
      - Top: map of sighted-ship wind force comparison (your existing formatting)
      - Bottom: two histograms (absolute differences): ΔBF and ΔWind Direction (22.5° buckets)
    Adds subplot labels: 'a' (map) and 'b' (hist row).

    Expects df_pairs columns:
      Longitude_1, Latitude_1, Wind Force_1
      Longitude_2, Latitude_2, Wind Force_2
      Bearing_1 / WD_Bearing_1, Bearing_2 / WD_Bearing_2  (for wind direction)
    """

    if df_pairs.empty:
        print("No pairs to plot.")
        return

    fig = plt.figure(figsize=(19, 20), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[3.0, 2.0])


    ax_map = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.coastlines(linewidth=0.5)
    ax_map.add_feature(cfeature.LAND, facecolor='#b0b0b0')

    # normalizer across both ships’ BF values
    vmin = np.nanmin([df_pairs['Wind Force_1'].min(), df_pairs['Wind Force_2'].min()])
    vmax = np.nanmax([df_pairs['Wind Force_1'].max(), df_pairs['Wind Force_2'].max()])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # gridlines
    gl = ax_map.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # plot ship 1 as circles (slight offset to reduce overlap)
    sc1 = ax_map.scatter(df_pairs['Longitude_1'] - 0.5, df_pairs['Latitude_1'] - 0.5,
                         c=df_pairs['Wind Force_1'], cmap='Blues', norm=norm,
                         marker='o', s=100, edgecolor='black', label='Ship 1',
                         alpha=0.8, transform=ccrs.PlateCarree())
    # plot ship 2 as triangles (slight offset)
    sc2 = ax_map.scatter(df_pairs['Longitude_2'] + 1.0, df_pairs['Latitude_2'] + 1.0,
                         c=df_pairs['Wind Force_2'], cmap='Blues', norm=norm,
                         marker='^', s=150, edgecolor='black', label='Ship 2',
                         alpha=0.8, transform=ccrs.PlateCarree())

    plt.suptitle("Mapped Wind Force Comparison Between Sighted Ships", fontsize=24, y=0.96)
    cbar = fig.colorbar(sc1, ax=ax_map, label="Beaufort Wind Force Value", fraction=0.025, pad=0.05)

    # set label font size
    cbar.set_label("Beaufort Wind Force Value", fontsize=20)
    
    # set tick label size
    cbar.ax.tick_params(labelsize=16)
    try:
        ticks = np.arange(int(vmin), int(vmax) + 1)
        cbar.set_ticks(ticks)
    except Exception:
        pass


    ax_map.legend()

    # subplot label 'a' (top-left of the map)
    ax_map.text(-0.01, 0.99, 'a', transform=ax_map.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='left')

    # Ensure absolute-difference series exist (compute on the fly if needed)
    def _pick_col(df: pd.DataFrame, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    d = df_pairs.copy()

    # BF_abs_diff
    if 'BF_abs_diff' not in d.columns:
        bf1_col = _pick_col(d, ['BF Value_1', 'Wind Force_1', 'BF_1', 'Beaufort_1'])
        bf2_col = _pick_col(d, ['BF Value_2', 'Wind Force_2', 'BF_2', 'Beaufort_2'])
        if bf1_col and bf2_col:
            bf1 = pd.to_numeric(d[bf1_col], errors='coerce')
            bf2 = pd.to_numeric(d[bf2_col], errors='coerce')
            valid = bf1.notna() & bf2.notna()
            d['BF_abs_diff'] = np.where(valid, np.abs(bf1 - bf2), np.nan)
        else:
            raise ValueError("Could not find BF columns to compute BF_abs_diff.")

    # WD_abs_diff_deg
    if 'WD_abs_diff_deg' not in d.columns:
        wd1_col = _pick_col(d, ['Bearing_1', 'WD_Bearing_1', 'Wind Direction_1'])
        wd2_col = _pick_col(d, ['Bearing_2', 'WD_Bearing_2', 'Wind Direction_2'])
        if wd1_col and wd2_col:
            wd1 = pd.to_numeric(d[wd1_col], errors='coerce')
            wd2 = pd.to_numeric(d[wd2_col], errors='coerce')
            valid = wd1.notna() & wd2.notna()
            wd_signed = (wd1 - wd2 + 180.0) % 360.0 - 180.0
            d['WD_abs_diff_deg'] = np.where(valid, np.abs(wd_signed), np.nan)
        else:
            raise ValueError("Could not find wind-direction columns to compute WD_abs_diff_deg.")

    # Series for plotting (drop NaNs)
    bf = pd.to_numeric(d['BF_abs_diff'], errors='coerce').dropna()
    wd = pd.to_numeric(d['WD_abs_diff_deg'], errors='coerce').dropna()

    # Bins: BF integer centers; WD at 22.5° buckets with full-width first/last bars
    bf_max = int(np.nanmax(bf)) if len(bf) else 0
    bf_bins = np.arange(-0.5, bf_max + 1.5, 1)
    bf_xticks = np.arange(0, bf_max + 1, 1)

    half = 11.25
    wd_edges   = np.arange(-half, 180 + half + 1e-9, 22.5)  # -11.25 .. 191.25
    wd_centers = np.arange(0, 180 + 22.5, 22.5)             # 0 .. 180

    # bottom axes
    ax_bf = fig.add_subplot(gs[1, 0])
    ax_wd = fig.add_subplot(gs[1, 1])

    mid_blue = plt.get_cmap('Blues')(0.7)

    # BF histogram (keep your titles/labels)
    ax_bf.hist(bf, bins=bf_bins, color=mid_blue, edgecolor='black', alpha=0.85)
    ax_bf.set_title("Difference in Wind Force Values", fontsize = 24, y=1.01)
    ax_bf.set_xlabel("Δ in Recorded Beaufort Values", fontsize = 20)
    ax_bf.set_ylabel("Counts", fontsize = 20)
    ax_bf.set_xticks(bf_xticks)

    # WD histogram (keep your titles/labels)
    ax_wd.hist(wd, bins=wd_edges, color=mid_blue, edgecolor='black', alpha=0.85)
    ax_wd.set_title("Difference in Wind Directions", fontsize = 24, y=1.01)
    ax_wd.set_xlim(-half, 180 + half)   # ensures 0° bin shows full width
    ax_wd.set_xticks(wd_centers)
    ax_wd.set_xlabel("Δ in Recorded Wind Directions (Degrees)", fontsize = 20)
    ax_wd.set_ylabel("Counts", fontsize = 20)
    ax_wd.grid(axis='x', linestyle=':', alpha=0.4)

    # histogram tick labels
    for ax in (ax_bf, ax_wd):
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=14)

    # subplot label 'b' (top-left of the histogram row)
    ax_bf.text(-0.04, 1.1, 'b', transform=ax_bf.transAxes,
               fontsize=24, fontweight='bold', va='top', ha='left')

    # finalize
    if save and figures_path:
        os.makedirs(figures_path, exist_ok=True)
        outpath = os.path.join(figures_path, filename)
        plt.savefig(outpath, dpi=300, bbox_inches='tight')

    plt.show()

def summarize_bf_wd_agreement(df_pairs: pd.DataFrame) -> dict:
    """
    Returns:
      {
        'bf': {'n': int, 'count_le1': int, 'pct_le1': float},
        'wd': {'n': int, 'count_le22_5': int, 'pct_le22_5': float,
               'count_le45': int, 'pct_le45': float}
      }
    """
    def _pick(df, names):
        for n in names:
            if n in df.columns: return n
        return None

    d = df_pairs.copy()

    # --- Beaufort absolute difference ---
    if 'BF_abs_diff' in d:
        bf = pd.to_numeric(d['BF_abs_diff'], errors='coerce')
    else:
        b1 = _pick(d, ['BF Value_1','Wind Force_1','BF_1','Beaufort_1'])
        b2 = _pick(d, ['BF Value_2','Wind Force_2','BF_2','Beaufort_2'])
        if not (b1 and b2):
            raise ValueError("BF columns not found.")
        bf1 = pd.to_numeric(d[b1], errors='coerce')
        bf2 = pd.to_numeric(d[b2], errors='coerce')
        bf  = (bf1 - bf2).abs()

    bf = bf.dropna()
    n_bf = int(bf.size)
    c_bf1 = int((bf <= 1).sum())
    p_bf1 = float(c_bf1 / n_bf * 100) if n_bf else float('nan')

    # --- Wind direction absolute difference (degrees) ---
    if 'WD_abs_diff_deg' in d:
        wd = pd.to_numeric(d['WD_abs_diff_deg'], errors='coerce')
    else:
        w1 = _pick(d, ['Bearing_1','WD_Bearing_1','Wind Direction_1'])
        w2 = _pick(d, ['Bearing_2','WD_Bearing_2','Wind Direction_2'])
        if not (w1 and w2):
            raise ValueError("Wind-direction columns not found.")
        wd1 = pd.to_numeric(d[w1], errors='coerce')
        wd2 = pd.to_numeric(d[w2], errors='coerce')
        # minimal absolute angular difference
        wd  = ((wd1 - wd2 + 180) % 360 - 180).abs()

    wd = wd.dropna()
    n_wd   = int(wd.size)
    c_22_5 = int((wd <= 22.5).sum())
    c_45   = int((wd <= 45.0).sum())
    p_22_5 = float(c_22_5 / n_wd * 100) if n_wd else float('nan')
    p_45   = float(c_45   / n_wd * 100) if n_wd else float('nan')

    return {
        'bf': {'n': n_bf, 'count_le1': c_bf1, 'pct_le1': p_bf1},
        'wd': {'n': n_wd, 'count_le22_5': c_22_5, 'pct_le22_5': p_22_5,
               'count_le45': c_45, 'pct_le45': p_45}
    }


