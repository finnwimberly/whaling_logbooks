#!/usr/bin/env python
# coding: utf-8

# ## Consolidated Infilling Script (Tiers 2, 3, and 4)

# ### Description
# This script combines the functionality of the previous Tier2, Tier3, and Tier4 scripts
# into a single, configurable workflow. It sequentially infills data gaps of increasing
# duration (from 1 day up to 5 days), saving the output and printing statistical
# comparisons at each major step.

# ### Imports
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import datetime, os
from cartopy import crs as ccrs
from cartopy import feature as cfeature
import seaborn as sns
from tabulate import tabulate
import sys
from matplotlib.colors import Normalize
from geopy.distance import geodesic
from scipy import stats
import statsmodels.api as sm

pd.options.display.max_columns = 50
print(f"Script started on {datetime.datetime.now().ctime()}")

### Core Functions

from geopy.distance import geodesic
import numpy as np

def infill_missing_data(df, columns_to_infill, days_missing, max_distance_km):
    """
    Infill missing Latitude/Longitude values for specified columns and gap duration.
    Adds 'gap_distance_km' for *all* examined gaps (accepted or rejected).
    For accepted gaps, also writes: 'Infilled', 'infill_days_missing', 'infill_type',
    and 'infill_distance_km'.
    """
    print(f"  - Infilling {days_missing}-day gaps for columns: {columns_to_infill}...")

    KM_PER_DEG_LAT = 111.32  # average km per degree latitude

    # Ensure columns exist
    if 'Infilled' not in df.columns:
        df['Infilled'] = False
    if 'gap_distance_km' not in df.columns:
        df['gap_distance_km'] = np.nan

    df_sorted = df.sort_values(by=['LogBook ID', 'Entry Date Time']).copy()
    df_sorted.reset_index(inplace=True)  # keep original index in 'index'
    window_size = days_missing + 2

    want_lat = 'Latitude_decimal' in columns_to_infill
    want_lon = 'Longitude_decimal' in columns_to_infill
    which = ('latlon' if (want_lat and want_lon) else
             'lat' if want_lat else
             'lon')

    for i in range(len(df_sorted) - (window_size - 1)):
        window = df_sorted.iloc[i : i + window_size]

        # single logbook only
        if window['LogBook ID'].nunique() != 1:
            continue

        # interior rows of this gap in original df
        interior_idx = window['index'].iloc[1:-1]

        # skip if any interior rows already infilled
        if df.loc[interior_idx, 'Infilled'].any():
            continue

        start_point = window.iloc[0]
        end_point   = window.iloc[-1]
        gap_to_fill = window.iloc[1:-1]

        # endpoints must have data in target cols
        if not (start_point[columns_to_infill].notna().all() and end_point[columns_to_infill].notna().all()):
            continue
        # interior must be missing in target cols
        if not gap_to_fill[columns_to_infill].isna().all().all():
            continue
        # and the "other" coordinate(s) must be present
        other_cols = [c for c in ['Latitude_decimal', 'Longitude_decimal'] if c not in columns_to_infill]
        if other_cols and not gap_to_fill[other_cols].notna().all().all():
            continue

        # dates must be consecutive
        date_diff = (end_point['Entry Date Time'].date() - start_point['Entry Date Time'].date()).days
        if date_diff != (days_missing + 1):
            continue

        # compute distance in km (span distance)
        lat1, lon1 = float(start_point['Latitude_decimal']), float(start_point['Longitude_decimal'])
        lat2, lon2 = float(end_point['Latitude_decimal']),   float(end_point['Longitude_decimal'])

        if which == 'latlon':
            distance_km = geodesic((lat1, lon1), (lat2, lon2)).kilometers
        elif which == 'lat':
            distance_km = abs(lat2 - lat1) * KM_PER_DEG_LAT
        else:  # 'lon'
            dlon = lon2 - lon1
            if dlon > 180:  dlon -= 360
            if dlon < -180: dlon += 360
            dlon = abs(dlon)
            phi = np.deg2rad(0.5 * (lat1 + lat2))
            km_per_deg_lon = KM_PER_DEG_LAT * np.cos(phi)
            km_per_deg_lon = float(km_per_deg_lon) if km_per_deg_lon > 1e-6 else 0.0
            distance_km = dlon * km_per_deg_lon

        # ALWAYS log the span distance for this gap (even if we won't infill)
        df.loc[interior_idx, 'gap_distance_km'] = float(distance_km) if not np.isnan(distance_km) else np.nan
        df.loc[interior_idx, 'gap_days_missing'] = days_missing
        df.loc[interior_idx, 'gap_type'] = which

        # threshold check – if it fails, we logged gap_distance_km and move on
        if np.isnan(distance_km) or distance_km > max_distance_km:
            continue  # skip infill

        # --- perform the infill ---
        selection_for_interp = window[['Latitude_decimal', 'Longitude_decimal']].copy()

        # handle antimeridian before interpolation
        if want_lon:
            lon_series = selection_for_interp['Longitude_decimal']
            if lon_series.iloc[0] < -170 and lon_series.iloc[-1] > 170:
                lon_series = lon_series.apply(lambda x: x if x <= 0 else x - 360)
            elif lon_series.iloc[0] > 170 and lon_series.iloc[-1] < -170:
                lon_series = lon_series.apply(lambda x: x if x >= 0 else x + 360)
            selection_for_interp['Longitude_decimal'] = lon_series

        interpolated_values = selection_for_interp.interpolate(method='linear')

        # restore lon to [-180, 180] if shifted
        if want_lon:
            if (selection_for_interp['Longitude_decimal'] < -180).any():
                interpolated_values['Longitude_decimal'] += 360
            elif (selection_for_interp['Longitude_decimal'] > 180).any():
                interpolated_values['Longitude_decimal'] -= 360

        update_values = interpolated_values.iloc[1:-1][columns_to_infill]
        for col in columns_to_infill:
            df.loc[interior_idx, col] = update_values[col].values

        # mark metadata for accepted infill
        df.loc[interior_idx, 'Infilled'] = True
        df.loc[interior_idx, 'infill_days_missing'] = days_missing
        df.loc[interior_idx, 'infill_type'] = which
        df.loc[interior_idx, 'infill_distance_km'] = float(distance_km)

    return df
    

def calculate_statistical_significance(series1, series2, name1="Original", name2="New"):
    """
    Performs and prints the results of the Mann-Whitney U and Kruskal-Wallis tests.
    """
    print(f"  - Running statistical tests for '{name1}' vs. '{name2}'...")
    
    s1_numeric = pd.to_numeric(series1, errors='coerce').dropna()
    s2_numeric = pd.to_numeric(series2, errors='coerce').dropna()

    if len(s1_numeric) == 0 or len(s2_numeric) == 0:
        print("    - Could not run tests, one or both series have no valid data.")
        return

    result_mannwhitneyu = stats.mannwhitneyu(s1_numeric, s2_numeric)
    print(f"    - Mann Whitney U Test: {result_mannwhitneyu}")

    result_kruskal = stats.kruskal(s1_numeric, s2_numeric)
    print(f"    - Kruskal-Wallis Test: {result_kruskal}")

def drop_cols_if_present(df_in, cols):
    df_out = df_in.copy()
    for c in cols:
        if c in df_out.columns:
            df_out.drop(columns=[c], inplace=True)
    return df_out
