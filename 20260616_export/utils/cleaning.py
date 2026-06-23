# I HAVE EXPORTED ALL FILES USED IN THE INITIAL DATA CLEANING (TIER1.IPYNB) TO 
# THIS FILE IN ORDER TO CREATE A MORE CONCISE WORKFLOW

# THE DATA CLEANING PROCESS CONSISTS OF 5 MAIN STEPS:
    # 1) REMOVING DUPLICATE ENTRIES
    # 2) STANDARDIZING COLUMNS WITH EXTRA TEXT AND MAPPING WIND TERMS TO BF SCALE
    # 3) CATCHING/CORRECTING COORDINATE ERRORS
    # 4) PLOTTING TO CATCH MISSED ERRORS
    # 5) CREATE FINAL PLOTS AND EXPORTING THE CLEANED DATASET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, os, sys
import math
from cartopy import crs as ccrs
from cartopy import feature as cfeature
# import seaborn as sns # not used in this script
#from tabulate import tabulate # not used in this script
#from matplotlib.colors import Normalize # not used in this script
import re
from IPython.display import clear_output
import datetime
import textwrap
import csv
from collections import defaultdict
import difflib
import os, math


# 1) CORRECTING DUPLICATES --------------------------------------------------------------------------

def correct_dups(df, dup_ids, log_path="duplicate_corrections_log.txt"):
    def log(msg, end="\n"):
        with open(log_path, "a") as f:
            f.write(msg + end)

    def show_context(df_subset, ids):
        print(f"\n Context for suspected duplicates {ids}:\n")
        cols = ['LogBook ID', 'ID', 'Entry Date', 'Latitude', 'Longitude', 'Weather']
        cols = [col for col in cols if col in df_subset.columns]
        print(df_subset[cols].to_string(index=False))

    def compare_rows(r1, r2, fields):
        def shorten(val): return textwrap.shorten(str(val), width=40, placeholder="...") if not pd.isnull(val) else ""
        print("\n Side-by-side comparison:\n")
        df = pd.DataFrame({
            'Field': fields,
            f"Entry {r1['ID']}": [shorten(r1[f]) for f in fields],
            f"Entry {r2['ID']}": [shorten(r2[f]) for f in fields]
        })
        print(df.to_string(index=False))

    def update_entry_date(df, idx, new_date_str):
        try:
            new_date = pd.to_datetime(new_date_str).date()
            df.at[idx, 'Entry Date'] = new_date

            if 'Entry Date Time' in df.columns:
                cur_time = pd.to_datetime(df.at[idx, 'Entry Date Time']).time()
                df.at[idx, 'Entry Date Time'] = datetime.datetime.combine(new_date, cur_time)

            if 'DateTime' in df.columns:
                if pd.isnull(df.at[idx, 'DateTime']):
                    df.at[idx, 'DateTime'] = datetime.datetime.combine(new_date, datetime.time(12, 0))
                else:
                    cur_time = pd.to_datetime(df.at[idx, 'DateTime']).time()
                    df.at[idx, 'DateTime'] = datetime.datetime.combine(new_date, cur_time)

            print(f"Updated Entry Date (and time fields) to {new_date}")
            return True
        except Exception as e:
            print(f"Error parsing date: {e}")
            return False

    if len(dup_ids) != 2:
        print("Please provide exactly two IDs.")
        return df

    df = df.copy().sort_values(by='ID').reset_index(drop=True)
    idxs = [df.index[df['ID'] == i].tolist() for i in dup_ids]
    idxs = [i[0] for i in idxs if i]

    if len(idxs) != 2:
        print(f"Could not find both IDs: {dup_ids}")
        return df

    row1, row2 = df.loc[idxs[0]], df.loc[idxs[1]]
    logbook_id = row1['LogBook ID']
    logbook_df = df[df['LogBook ID'] == logbook_id].reset_index(drop=True)
    lb_idxs = [logbook_df.index[logbook_df['ID'] == i].tolist()[0] for i in dup_ids]
    start, end = max(0, min(lb_idxs) - 2), min(len(logbook_df), max(lb_idxs) + 3)
    show_context(logbook_df.iloc[start:end], dup_ids)

    key_fields = [col for col in [
        'ID', 'Entry Date', 'Latitude', 'Longitude', 'Weather',
        'Wind Direction', 'Wind Speed/Force', 'Ship Sightings',
        'Miscellaneous Observations', 'Cloud Cover', 'Sea State',
        'Bottom', 'Landmark', 'Ship Heading/Course', 'Page', 'Depth'
    ] if col in df.columns]

    compare_rows(row1, row2, key_fields)

    sys.stdout.flush()
    choice = input(
        f"\nWhat would you like to do?\n"
        f"[1]: Drop entry {row1['ID']}\n"
        f"[2]: Drop entry {row2['ID']}\n"
        f"[e]: Edit one of them\n"
        f"[n]: No changes\n"
        f"Your choice: "
    ).strip().lower()

    log(f"\n==== {datetime.datetime.now()} ====")
    log(f"Compared IDs: {row1['ID']} vs {row2['ID']}")

    if choice == '1':
        df = df[df['ID'] != row1['ID']].reset_index(drop=True)
        log(f"Action: Dropped {row1['ID']}")
    elif choice == '2':
        df = df[df['ID'] != row2['ID']].reset_index(drop=True)
        log(f"Action: Dropped {row2['ID']}")
    elif choice == 'e':
        edit_id = input("Enter the ID to edit: ").strip()
        if edit_id in [str(row1['ID']), str(row2['ID'])]:
            idx = df.index[df['ID'] == int(edit_id)][0]
            print("\nCurrent values:")
            for col in key_fields:
                print(f"{col}: {df.at[idx, col]}")
            new_date = input("\nNew Entry Date (YYYY-MM-DD): ").strip()
            if new_date and update_entry_date(df, idx, new_date):
                log(f"Action: Edited {edit_id}")
                log(f" - Updated Entry Date to: {new_date}")
            else:
                log(f"Failed to edit {edit_id}")
        else:
            print("Invalid ID.")
    else:
        print("No changes made.")
        log("Action: No changes made")
        comment = input("Comment (optional): ").strip()
        if comment:
            log(f"Comment: {comment}")

    log("Key field differences:")
    for field in key_fields:
        val1, val2 = row1[field], row2[field]
        if not (pd.isnull(val1) and pd.isnull(val2)) and val1 != val2:
            log(f" - {field}: {val1}  ≠  {val2}")

    return df

# 2) TEXT  FORMATTING / WIND MAPPING -----------------------------------------------------------

def clean_page_column(df, column="Page"):
    """
    Exact-match page cleaner mirroring clean_depth_column, with inline dicts.
    """
    df[f"{column}_og"] = df[column].copy()

    page_map = {
        '1-8': 1, '3 1/2': 3, '5 1/2': 5, '14-15': 14, '22-23': 22,
        '30-31': 30, '48-49': 48, '94/95': 94, '97-98': 97, '108-109': 108,
        '121 (says 107)': 121, '122-123': 122, '158-159': 158,
        '159-160': 160, '177-178': 177, '186-187': 186,
        '(8)': 8, '(6)': 6, '(4)': 4, '(2)': 2, '(16': 16,
        '(17': 17, '(18': 18, '(19': 19, '(20': 20, '49-52':49, '5 1/2a':5,
        '91A':91, '91B':91, '94-95':94, '96-97':96,
        '20-21': 20, '24-25': 24, '29-30': 29, '35-36': 35,
        }
    page_nan = {'N', 'N/a', 'n/a', 'n/a`', '(', '1870-05-08'}

    # apply conversions
    df[column] = df[column].replace(page_map)

    # set NaN for flagged originals
    df.loc[df[f"{column}_og"].isin(page_nan), column] = np.nan

    # light trim + numeric
    df[column] = df[column].astype(str).str.strip('"ab` ').replace({"nan": np.nan})
    df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")

    # report unhandled
    unhandled = df[df[column].isna() & df[f"{column}_og"].notna()][f"{column}_og"]
    unknown = sorted(set(unhandled) - set(page_map.keys()) - set(page_nan))
    if unknown:
        print(f"\n{len(unknown)} unhandled values (add to page_map or page_nan):")
        for val in unknown:
            print(f"  - '{val}'")
    else:
        print("All page values successfully handled")

    return df


def clean_depth_column(df, column='Depth'):
    """
    Clean and standardize the Depth column in the dataframe using exact matches only.
    Returns a DataFrame with:
      - 'Depth_og': original string value
      - 'Depth': cleaned float or NaN
      - Console output of any unhandled strings (not in depth_map or depth_nan)
    """

    # save original values
    df['Depth_og'] = df[column].copy()

    # running list of conversions
    depth_map = {
        'anchored in 5 fathoms of water': 5,
        '2 1/2': 2.5, '3 1/2': 3.5, '4 1/2': 4.5, '5 1/4': 5.25,
        '5 1/2': 5.5, '6 1/2': 6.5, '7 1/2': 7.5, '8 1/2': 8.5, '8.5': 8.5,
        '9 1/2': 9.5, '11 1/2': 11.5, '12 1/2': 12.5,
        '8 (at Tarpaulin Cove)': 8.0,
        '30 and 25': 27.5,
        '35(11pm), 38(2am), 33(11am)': 35.0,
        '70 at 5': 70.0,
        'at 11 pm sounded in 45': 45.0,
        '23 @ 8:30pm, then 10 @ noon': 16.5,
        '45 in am, 30 at noon (end of day)': 38.0,
        'at 8am 26': 26.0,
        '@ 8pm 7': 7.0, '@ 4pm 45': 45.0, '@ 4pm 90': 90.0,
        "'@ 4pm 90": 90.0, "'@ 4pm 45": 45.0, "'@ 8pm 7": 7.0,
        '48 at 3:30pm, 45 at 5pm': 47.0,
        '7-9': 8.0,
        '5 @ 2': 5.0, '5 @ 2am': 5.0,
        '20 (at 4pm)': 20.0,
        '@5pm-22': 22.0, "'@5pm-22": 22.0,
        'at 3pm 4': 4.0, 'at 9pm 23': 23.0, 'at 11am 50': 50.0,
        '3pm-20': 20.0, '4.5 @ 1pm': 4.5,
        '10 @ 7pm': 10.0,
        '28(8pm), 30(rest of night)': 29.0,
        '5 Fathoms' : 5.0,
        '6 to 3 (due to storm) then back to 6' : 6.0,
        '45 at 4pm' : 45.0,
        '60 fathoms' : 60,
        '27 & 30': 28.5,
        '55?': 55,
        '6-8': 7,
        '@ 3am 58, @ 7am 52': 55,
        'at 4pm 80, at 6pm 52': 66,
    }

    # running list of range/unclear values to set to NaN
    depth_nan = {
        'nan', '10-15', '20 then 11', '50 at start, 43 at 4 PM', '37, 27',
        '35, 30', '50, 45, 40 (7pm, 11pm, 4am)', '35, 22 (1pm, 4pm)',
        '54 (8pm), 70 (2am), 90 (4am)', '50 (7pm), 65 (noon)', "'@ 3am 58, @ 7am 52",
        '`', '7pm 41, 9pm 31', '70 at 5am', '19 at 1:30pm, 17 at 7pm, 9 at 5am',
        '35 @ 7pm, 26 @ 8am, 20 @ 12pm', '41 to 75', '11, 4', 'ENE',
        '9am-17, 11am-20', '50 @ 10pm, 30 later',
        '50 @ 2pm, 35 @ 11pm, 28 @ 3am, 42 @ 9am',
        '20 @ 11pm, 10 @ 8am',
        '48, 27, 30 @ 10am', '22 38 W'
    }

    # apply conversions
    df[column] = df[column].replace(depth_map)

    # set depth_nan entries to NaN
    df.loc[df['Depth_og'].isin(depth_nan), column] = np.nan

    # convert strings to floats
    df[column] = pd.to_numeric(df[column], errors='coerce')

    # print strings not found in dicts that errored
    unhandled = df[df[column].isna() & df['Depth_og'].notna()]
    unknown = sorted(set(unhandled['Depth_og']) - depth_nan - depth_map.keys())
    if unknown:
        print(f"\n{len(unknown)} unhandled values (add to depth_map or depth_nan):")
        for val in unknown:
            print(f"  - '{val}'")
    else:
        print("All depth values successfully handled")

    return df

def clean_wind_dirs(df):
    """
    Clean wind direction data in a DataFrame column named 'Wind Direction'
    
    Args:
        df (pd.DataFrame): DataFrame with 'Wind Direction' column
        
    Returns:
        tuple: (cleaned_df, removed)
            - cleaned_df: DataFrame with cleaned 'Wind Direction' column
            - removed: list of terms that were set to NaN
    """
    
    # Make a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Define valid wind directions
    valid_WD = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    
    # These are patterns that indicate the entire entry should be set to NaN
    removal_patterns = [
        'BAFFLING/VARIABLE', 'BAFFLING/VARIABLE,', 'BAFFLING', 
        'BAFFLING/VARIBLE', 'BAFFLING/VARIBALE', 'VERY BAFFLING',
        'VARIABLE', 'VARIABLE', 'VARIABLES', 'VARIABLES',
        'AND HAULING', '@', '[?; NONE RECORDED]', '[?;ONEECORDED]', '& VARIABLE', 
        'HEAD WIND', 'STRONG',  'ALL WAS', 'ALL WAS OF COMPASS', 'MODERATING', 'LIGHT', 
        'LIGHT AIRS', 'LIGHT VARIABLE WIND', 'PASSING SQUALLS', 'SQUALLY', 'CONSTANTLY VEERING', 
        'GOING DOWN THE BAY', 'CHANGEABLE', 'VARIOUS', 'VARIOUS', 'STRONG', 'ALL', 
        'THE LAND', 'WINDS', 'SEA', 'THE SEA', 'AND AND SEA', 'AM LAND & SEA', 'AM SEA', 
        'AM SEA & LAND', 'SEA & LAND', 'TABLE MOUNTAIN', 'TO LEEWARD', 'SHIFTING TRADES',
        'OFF SHORE', 'HT WIND', 'NE (HAULED SW 2PM)', 'BAFFLING/VARIABLE FROM NEW TO NE',
        'NE TO SW', 'WSN', 'NSW', 'EW TO WNW', 'NNE @ SSE', 'N TO SW', 'NEW',
        'ALL AROUND THE COMPASS', 'ALL AROUND COMPASS', 'ALL AROUND THE COMPASS', 'ALL AROUND',
        'ALL QUARTERS', 'ALL POINTS', 'ALL POINTS OF THE COMPASS', 'ALL POINTS ON THE COMPASS',
        'ALL CORNERS', 'ALL QUARTERS OF THE COMPASS', 'ALL WAYS', 'ALL WAYS OF THE COMPASS', 
        'ALL DIRECTIONS FROM THE COMPASS', 'ALL DIRECTIONS OF THE COMPASS', 'ALL DIRECTIONS',
        'ALL DIRECTIONS', 'ALL POINTS OF THE COMPASS', 'ALL PARTS OF THE COMPASS', 
        'ALL PARTS OF THE COMPASS', '"ALL ROUND THE HOUSE"', 'ALL ROUND THE COMPASS', 
        'ALL ROUND THE COMPASS', 'ALL POINTS',
        '"THE TRADES VARYING ABOUT 2 POINTS EACH WAY"', 'VARYING', 'VARAIBLE', 
        'CALMS', 'CALM', 'FROM SEA & LAND', 'ALL QUARTERS OF THE COMPASS', 'FROM ALL PARTS OF THE COMPASS',
        'CONSTANTLY VEERING', 'ALL AROUND COMPASS', 'ALL QUARTERS', 'VARIABLES', 'ALL POINTS OF THE COMPASS',
        'FRAM SEA & LAND', 'ALL ROUND THE COMPASS', 'GOING DOWN THE BAY', 'PARTS OF COMPASS', 'POINTS', 'POINTS OF COMPASS', 
        'POINTS ON COMPASS', 'QUARTERS', 'QUARTERS OF COMPASS', 'ROUND COMPASS', 'POINTS OF COMPASS'
    ]
    
    # # Define additive terms to convert to "TO"
    # additive_terms = ['AND', '&', 'BY', 'NY', 'HALF', 'OR', '/', 'INTO', '+', 'HEADING']
    
    # Eastward equivalences - ALL CAPS
    east_patterns = {
        'EASTERLY': 'E', 'EAST': 'E', 'EASTWARD': 'E', 'EWARD': 'E', 'EE': 'E',
        'MOSTLY FROM THE E': 'E', '" OFF SHE" W TO EAST': 'E',
        'ASTERLY': 'E', 'E OF E': 'E', 'BREEZED UP FROM E': 'E', '" OFF SHORE" W HEADING EAST': 'E',
        '"FROM OFF SHORE", FROM THE W HEADING EAST': 'E', 'E+SE' : 'ESE'
    }

    # Westward equivalences - ALL CAPS
    west_patterns = {
        'WESTERLY': 'W', 'WEST': 'W', 'WESTWARD': 'W', 'WWARD': 'W',
        'BAFFLING/VARIABLE, FROM THE WEST': 'W',
        'WN': 'W', 'SW NW AFFLING': 'W', 'WESTWARD': 'W',
        'WEST WESTWARD': 'W', 'W WESTWARD': 'W'
    }

    # Southward equivalences - ALL CAPS
    south_patterns = {
        'SOUTH': 'S', 'SOUTHWARD': 'S', 'SWARD': 'S', 'S HEADING INTO EASTWARD': 'SSE',
        'S HEADING INTO E': 'SSE', 'MOSTLY FROM THE S': 'S', 'SS': 'S', 'S(?)': 'S',
        'SS(?)': 'S', 'SOUTH WEST': 'SW', 'S WEST': 'SW', 'S WESTWARD': 'SW',
        'SOUTHWEST': 'SW', 'SOUTHEAST': 'SE', 'SOUTH EAST': 'SE', 'SOUTH E': 'SE',
        'S EAST': 'SE', 'S EASTWARD': 'SE', 'SE': 'SE', 'SES' : 'SSE',
        'S TO SSW UNDER ALL SAIL' : 'S', 'S&W' : 'SW', 'SSE BY THE WINDS' : 'SSE', 'SE BY ESE' 
        : 'SE TO ESE', 'S BY ESE' : 'SSE', 'BAFFLING S TO SE' : 'SSE', 'SOTHERLY': 'S'
    }

    # Northward equivalences - ALL CAPS
    north_patterns = {
        'NORTH': 'N', 'NORTHWARD': 'N', 'NWARD': 'N', 'NN': 'N', 
        'NORTH WEST': 'NW', 'N WEST': 'NW', 'NORTHWEST': 'NW', 'N WESTWARD': 'NW',
        'NORTH EAST': 'NE', 'N EAST': 'NE', 'N EASTWARD': 'NE', 'NORTHWARS': 'N',
        'NORTHWARD AND EASTWARD': 'NE', 'NE*' :'NE', 'N, THEN NW' : 'NNW',
        'ORTHWARD' : 'N', 'N&W' : 'NW', 'N&E' : 'NE', 'N BY ENE' : 'NNE', 'N BY NNE' : 'NNE',
        'NORTHER' : 'N', 'NORTHERLY': 'N'
    }

    # Compound direction patterns - ALL CAPS
    compound_patterns = {
        'EN': 'NE', 'NEE': 'ENE', 'NER': 'NE', 'N+E' : 'NE',
        'BAFFLING/VARIABLE FROM THE NE': 'SW', 
        'NE NE': 'NE', 'NENE': 'NE', 'NEN': 'NNE', 'NE': 'NE', 'NR': 'N',
        'MOSTLY FROM THE NE': 'NE', 'E N': 'NE', 'N HAULS TO E': 'NE',
        'E+N': 'NE', 'NEN': 'NNE', 'NEE': 'ENE', 'E N E': 'ENE', 'ENN': 'NE',
        'SE': 'SE', 'ES': 'SE', 'E S': 'SE', 'E+S': 'SE', 'S+E' : 'SE',
        'SE OR THEREABOUT': 'SE', 'S EASTWARD': 'SE', 
        'WS': 'SW', 'SWW': 'SW', 'SWSW': 'SW', 'SW': 'SW', 'W+S': 'SW',
        'SSW': 'SSW',  'SWS': 'SSW', 
        'NWN': 'NNW', 'NNW': 'NNW', 'NWW': 'WNW', 'NNE': 'NNE', 'NE N': 'NNE',
        'ES3': 'ESE', 'ESE': 'ESE', 'SEE': 'ESE', 'SSSE': 'SSE', 'SE3': 'SSE',
        'WN': 'NW', 'W+N': 'NW', 'N+W':'NW', 'S+W' : 'SW', 'S, HALLED TO SE' : 'SSE',
        # Handle strange combinations
        'NSEW': np.nan,  # All directions - remove
        'NNN': 'N', 'SSN': np.nan, 'NSN': np.nan,  
        'WWW': 'W', 'WW': 'W', 
        'NE1': 'NE', 'SE1': 'SE', 'ENE1': 'ENE',  # Remove the '1'
        'SE?': 'SE',  # Remove the '?'
        'W NW': 'W TO NW',  
        'N W': 'NW',  
        'S SW': 'SSW', 
        'N W BY W': 'WNW', 
        'N TO NE AND E': 'NE', 
        'N BY E AND N': 'NNE',
        'E, S' : 'SE',
        'NNW BY N' : 'NNW',
        'ESS' : 'SSE'
    }
    
    # Track uncleaned terms
    uncleaned_terms = set()
    
    # pre-sort direction dictionaries
    east   = dict(sorted(east_patterns.items(),   key=lambda kv:-len(kv[0])))
    west   = dict(sorted(west_patterns.items(),   key=lambda kv:-len(kv[0])))
    south  = dict(sorted(south_patterns.items(),  key=lambda kv:-len(kv[0])))
    north  = dict(sorted(north_patterns.items(),  key=lambda kv:-len(kv[0])))
    comp   = dict(sorted(compound_patterns.items(), key=lambda kv:-len(kv[0])))

    dirs_lookup = (east, west, south, north, comp)   # ordered search list

    def _clean_one(raw):
        if pd.isna(raw) or raw == '':
            return np.nan
    
        # Pre-normalize
        s = str(raw).upper().strip()
        s = re.sub(r'\s+', ' ', s)
    
        # Check for full-string removal
        if s in removal_patterns:
            return np.nan
    
        # Check for full-string directional mapping
        for d in dirs_lookup:
            if s in d:
                mapped = d[s]
                # If mapping says np.nan, remove
                if mapped is None or (isinstance(mapped, float) and np.isnan(mapped)):
                    return np.nan
                return mapped
    
        s = re.sub(r'^(FROM|INCLINING|INCLINED)\s+', '', s)
    
        # Remove variable/baffling phrases
        removal_regexes = [re.compile(rf'\b{re.escape(p)}\b') for p in removal_patterns]
        if any(r.search(s) for r in removal_regexes):
            return np.nan
    
        # Find patterns like "SW by S" or "E by N"
        # Convert as: E by N to E SE  --> NE TO ESE
        pattern_by = re.compile(
            r'\b(' + '|'.join(valid_WD) + r')\s+BY\s+(' + '|'.join(valid_WD) + r')\b'
        )
    
        def replace_by(match):
            first = match.group(1)
            second = match.group(2)
            combined = second + first  # e.g. S + SW = SSW
            if combined in valid_WD:
                return combined
            # Try with no duplicate letters (e.g. S + S = S)
            return second + first
    
        # Replace all by-patterns (could be more than one per string)
        s = pattern_by.sub(replace_by, s)
    
        # Now handle the rest: replace conjunctions with 'TO' but skip "BY"
        s = re.sub(r'\b(?:AND|NY|HALF|OR|/|INTO|HEADING)\b', ' TO ', s)
        s = re.sub(r'(\w)[+&](\w)', r'\1 TO \2', s)
        tokens = [tok.strip() for tok in re.split(r'\s+', s) if tok.strip()]
    
        # Map each token using all dictionaries
        def map_token(tok):
            if tok == 'TO':
                return tok
            for d in dirs_lookup:
                if tok in d:
                    return d[tok]
            return tok
        mapped = [map_token(tok) for tok in tokens]
    
        # Split into groups by 'TO'
        groups = []
        group = []
        for tok in mapped:
            if tok == 'TO':
                if group:
                    groups.append(group)
                    group = []
                groups.append(['TO'])
            else:
                group.append(tok)
        if group:
            groups.append(group)
    
        # Merge tokens within each group to valid directions (3-letter, then 2-letter, then 1)
        def collapse_group(g):
            if len(g) == 3:
                tri = ''.join(g)
                if tri in valid_WD:
                    return [tri]
            if len(g) == 2:
                duo = ''.join(g)
                if duo in valid_WD:
                    return [duo]
            return [t for t in g if t in valid_WD]
        collapsed = []
        for gr in groups:
            if gr == ['TO']:
                collapsed.append('TO')
            else:
                collapsed += collapse_group(gr)
    
        # Remove stray 'TO's at the beginning or end
        while collapsed and collapsed[0] == 'TO':
            collapsed = collapsed[1:]
        while collapsed and collapsed[-1] == 'TO':
            collapsed = collapsed[:-1]
        # Remove consecutive 'TO's
        res = []
        for t in collapsed:
            if not (res and t == 'TO' and res[-1] == 'TO'):
                res.append(t)
        collapsed = res
    
        # If empty or only TO, return NaN
        if not collapsed or all(t == 'TO' for t in collapsed):
            return np.nan
    
        # Build the output string, joining groups only if needed
        output = []
        i = 0
        while i < len(collapsed):
            if collapsed[i] == 'TO':
                output.append('TO')
                i += 1
            else:
                # Gather consecutive directions until next 'TO'
                group = []
                while i < len(collapsed) and collapsed[i] != 'TO':
                    group.append(collapsed[i])
                    i += 1
                if len(group) > 1:
                    output.append(' TO '.join(group))
                else:
                    output.extend(group)
        # Now join output, making sure not to add redundant 'TO'
        cleaned_str = []
        for idx, part in enumerate(output):
            if idx > 0 and part != 'TO' and output[idx-1] != 'TO':
                cleaned_str.append('TO')
            cleaned_str.append(part)
        return ' '.join(cleaned_str)


    # Track original values to determine leftovers later
    original_wd = cleaned_df['Wind Direction'].copy()

    # apply cleaning
    cleaned_df['Clean Wind Direction'] = cleaned_df['Wind Direction'].apply(_clean_one)

    # leftovers are original entries whose cleaned value is NaN
    removed = (original_wd[cleaned_df['Clean Wind Direction'].isna()]
                 .dropna()
                 .unique()
                 .tolist())

    return cleaned_df, removed

def wind_dir_to_numeric(df, col='Clean Wind Direction', out_col='WD_Bearing'):
    """Convert cleaned wind-direction strings to numeric bearings (0–360) 
    and add calculated value to df in new column.

    Rules
    -----
    • Single directions to standard bearing (N=0, E=90, etc.)
    • Composite strings like "N TO NE TO E" use circular mean of all tokens.
    • If the largest angular span between min & max tokens exceeds 90°
      the value is set to Nan (per spec).
    • NaNs remain NaN.

    Parameters
    ----------
    df  : pd.DataFrame   – must already contain *cleaned* values from in *col*.
    col : str            – column name with cleaned strings.

    Returns
    -------
    pd.Series of float   – bearings in degrees.
    """

    # mapping table (degrees clockwise from North)
    deg_map = {
        'N': 0.0,  'NNE': 22.5,  'NE': 45.0,  'ENE': 67.5,
        'E': 90.0, 'ESE': 112.5, 'SE': 135.0, 'SSE': 157.5,
        'S': 180.0,'SSW': 202.5, 'SW': 225.0, 'WSW': 247.5,
        'W': 270.0,'WNW': 292.5, 'NW': 315.0, 'NNW': 337.5
    }

    def _numeric(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().upper()
        # fast path – single direction
        if val in deg_map:
            return deg_map[val]

        if ' TO ' not in val:
            return np.nan   # unknown token

        parts = [p.strip() for p in val.split(' TO ') if p.strip()]
        try:
            angles = [deg_map[p] for p in parts]
        except KeyError:
            return np.nan  # unexpected token inside composite

        if not angles:
            return np.nan

        # compute smallest angular span
        ang_sorted = sorted(angles)
        span1 = ang_sorted[-1] - ang_sorted[0]
        span2 = (ang_sorted[0] + 360) - ang_sorted[-1]
        largest_span = max(span1, span2)
        if largest_span > 90:
            return np.nan  # exceeds allowed span

        # circular mean
        radians = np.radians(angles)
        x = np.cos(radians).mean()
        y = np.sin(radians).mean()
        bearing = (np.degrees(np.arctan2(y, x)) + 360) % 360
        return bearing

    # Compute new bearings as a Series
    bearings = df[col].apply(_numeric)

    # Find where "Wind Direction" column is
    col_idx = df.columns.get_loc(col)
    # Insert new column right after "Wind Direction"
    df_out = df.copy()
    df_out.insert(col_idx + 1, out_col, bearings)

    return df_out

def init_wind_force_clean(df, col='Wind Speed/Force'):
    """
    Correct common errors in wind speed/force strings.
    Returns a copy of the DataFrame with the cleaned column.
    """
    cleaned_df = df.copy()
    
    # Lowercase everything
    cleaned_df[col] = cleaned_df[col].str.lower()
    
    # Set certain string values to NaN
    for st in ['from ne', 'sw']:
        cleaned_df.loc[cleaned_df[col] == st, col] = np.nan
    
    # Replace 'widns', 'windq', '"wind"' with 'winds'
    for word in ['widns', 'windq', '"wind"']:
        cleaned_df[col] = cleaned_df[col].str.replace(word, 'winds')
    
    # Replace 'breezs', 'breeeze', 'breezesd' with 'breezes'
    for word in ['breezs', 'breeeze', 'breezesd']:
        cleaned_df[col] = cleaned_df[col].str.replace(word, 'breezes')
    
    # Replace additional strings
    cleaned_df[col] = cleaned_df[col].str.replace('fne', 'fine')
    cleaned_df[col] = cleaned_df[col].str.replace('string', 'strong')
    cleaned_df[col] = cleaned_df[col].str.replace('aires', 'airs')
    cleaned_df[col] = cleaned_df[col].str.replace('light light winds', 'light winds')
    
    # Special case: 'light bafflin' -> 'light baffling'
    cleaned_df.loc[cleaned_df[col] == 'light bafflin', col] = 'light baffling'
    
    # Replace 'baffling' variants
    for st in ['"baffling"', 'blaffling', 'bafling', 'baflin']:
        cleaned_df[col] = cleaned_df[col].str.replace(st, 'baffling')
    
    # Convert to plural if endswith 'wind' or 'breeze'
    for st in ['wind', 'breeze']:
        endswith_mask = cleaned_df[col].str.endswith(st)
        mask = endswith_mask.where(pd.notna(endswith_mask), False)
        cleaned_df.loc[mask, col] = cleaned_df.loc[mask, col].str.replace(st, st + 's', regex=False)
    
    return cleaned_df


def load_beaufort_map(filename, unique_only=False):
    bf_map = defaultdict(list)  # Keep a list for order if unique_only=True
    current_bf = None
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('## Beaufort '):
                suf = line.split('## Beaufort ')[1]
                if suf == 'nan':
                    current_bf = np.nan
                elif '.' in suf:
                    current_bf = float(suf)
                else:
                    current_bf = float(suf)
            elif line:
                term = line.lower()
                if current_bf not in bf_map[term]:
                    bf_map[term].append(current_bf)

    if unique_only:
        # Only keep the first value for each term
        bf_map_flat = {k: v[0] for k, v in bf_map.items() if v}
    else:
        # Original: keep sets or flatten if only one
        bf_map_flat = {k: (v[0] if len(v) == 1 else set(v)) for k, v in bf_map.items() if v}
    return bf_map_flat

def save_beaufort_map(bf_map, filename):
    # Invert: category -> list of terms

    section_map = defaultdict(list)
    for term, vals in bf_map.items():
        if isinstance(vals, (set, list, tuple)):
            for v in vals:
                section_map[v].append(term)
        else:
            section_map[vals].append(term)

    # Define sortkey BEFORE use
    def sortkey(val):
        if pd.isna(val):
            return 1e10
        return float(val)

    with open(filename, 'w', encoding='utf-8') as f:
        for cat in sorted(section_map, key=sortkey):
            if pd.isna(cat):
                suf = "nan"
            elif isinstance(cat, float) and cat.is_integer():
                suf = str(int(cat))
            else:
                suf = str(cat)
            f.write(f"## Beaufort {suf}\n")
            for term in sorted(set(section_map[cat])):
                f.write(f"{term}\n")
            f.write("\n")

def parse_beaufort_series(
    df, col, bf_map, new_col='BF Value',
    mapping_txt_file=None, interactive=True, log_file=None
):
    """
    Assign a Beaufort value to each term in the specified column,
    using bf_map as the lookup dict. If not found, suggest closest matches.
    If interactive=True, prompt the user to confirm or enter new category.
    Optionally updates mapping_txt_file and logs additions to log_file.
    """
    results = []
    unknown_terms = set()

    # First pass: collect unique unknown terms for counting and progress tracking
    for val in df[col]:
        v = val if pd.isna(val) else str(val).strip().lower()
        if pd.isna(v) or v == '' or (v in bf_map and pd.isna(bf_map[v])):
            continue
        elif v in bf_map:
            continue
        else:
            unknown_terms.add(v)
    unknown_terms = sorted(unknown_terms)
    num_unknowns = len(unknown_terms)

    term_answers = {}  # Cache user responses for each new unknown term
    new_terms_to_write = {}

    # Print instructions once at start
    showed_instructions = False
    unknown_counter = 1  # Start counter at 1

    for val in df[col]:
        v = val if pd.isna(val) else str(val).strip().lower()
        if pd.isna(v) or v == '' or (v in bf_map and pd.isna(bf_map[v])):
            results.append(np.nan)
        elif v in bf_map:
            results.append(bf_map[v])
        elif v in term_answers:
            results.append(term_answers[v])
        else:
            # Print instructions before the first new term prompt
            if not showed_instructions and interactive:
                print("\nINSTRUCTIONS:")
                print("- When prompted for an unknown wind force term, you may:")
                print("    • Press Enter to accept the suggested BF value (if given),")
                print("    • Type a number (e.g., 0, 2.5, 8), or")
                print("    • Type 'nan' (without quotes) if the value is unknown or unclassifiable.")
                print("- Your classification will be added to the mapping text file for future use (if enabled).")
                print(f"\nTotal unique unknown wind force terms to classify: {num_unknowns}\n")
                showed_instructions = True

            # Show which number of unknown is being classified (based on order in unique list)
            if interactive and num_unknowns > 1:
                print(f"\nUnknown term {unknown_counter} of {num_unknowns}:")
            # Prompt as before
            close = difflib.get_close_matches(v, list(bf_map.keys()), n=3, cutoff=0.7)
            if close:
                suggestion = close[0]
                suggested_val = bf_map[suggestion]
                print(f"\nUnknown term: '{v}'")
                print(f"Suggested closest match: '{suggestion}' (BF Value: {suggested_val})")
                print("Other suggestions:")
                for term in close:
                    print(f"  - {term} (BF Value: {bf_map[term]})")
                user_input = input(f"Assign BF value for '{v}' [press Enter to accept {suggested_val}, or type new value]: ").strip()
                if user_input == "":
                    bf_value = suggested_val
                elif user_input.lower() == "nan":
                    bf_value = np.nan
                else:
                    try:
                        bf_value = float(user_input)
                    except:
                        bf_value = np.nan
                results.append(bf_value)
                term_answers[v] = bf_value
                new_terms_to_write[v] = bf_value
            else:
                print(f"\nUnknown term: '{v}'")
                user_input = input(f"No close match found. Please enter BF value for '{v}': ").strip()
                if user_input.lower() == "nan":
                    bf_value = np.nan
                else:
                    try:
                        bf_value = float(user_input)
                    except:
                        bf_value = np.nan
                results.append(bf_value)
                term_answers[v] = bf_value
                new_terms_to_write[v] = bf_value

            unknown_counter += 1  # increment for each unique unknown handled

    # Add to new column
    outdf = df.copy()
    outdf[new_col] = results

    # If new terms were added and we want to update the txt file
    if mapping_txt_file and new_terms_to_write:
        # Update bf_map with new terms
        for k, v in new_terms_to_write.items():
            bf_map[k] = v
        # Save updated map using your save_beaufort_map function
        save_beaufort_map(bf_map, mapping_txt_file)

    # Log file of additions (optional)
    if log_file and new_terms_to_write:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a", encoding='utf-8') as logf:
            logf.write(f"==== Beaufort additions {now_str} ====\n")
            for term, val in new_terms_to_write.items():
                val_str = "nan" if pd.isna(val) else str(val)
                logf.write(f"{term}: {val_str}\n")
            logf.write("\n")

    return outdf

def clean_remaining_strings(df):
    """Standardize 'Sea State', 'Cloud Cover', and 'Weather' columns in place."""
    # Sea State standard replacements
    replace_dict = {
        '"A Big Swell going"': 'Big Swell',
        'Big swell going': 'Big Swell',
        'Rough, running under topsails': 'Rough',
        'Rough Sea': 'Rough Seas',
        'Heavy Sea': 'Heavy Seas',
    }
    df['Sea State'] = df['Sea State'].replace(replace_dict)

    for st in ['Calm', 'Calms']:
        df['Sea State'] = df['Sea State'].str.replace(st, 'calm', regex=False)

    # Lower-case key phrases in 'Sea State'
    for st in ['Heavy', 'Swell', 'Bad', 'Very', 'Large', 'Rough', 'Rugged', 'Big', 'High', 'Water', 'Seas', 'Sea',
               'Running', 'Remarkably', 'Considerable', 'Heaving', 'Moderate', 'Pleasant', 'Smooth', 'Unsettled']:
        contains_mask = df['Sea State'].str.contains(st, na=False)
        df.loc[contains_mask, 'Sea State'] = df.loc[contains_mask, 'Sea State'].str.replace(st, st.lower(), regex=False)

    # Clean 'Cloud Cover'
    df['Cloud Cover'] = df['Cloud Cover'].str.lower().str.replace('smokey', 'smoky', regex=False)

    # Clean 'Weather'
    df['Weather'] = (
        df['Weather']
        .str.lower()
        .str.replace('"', '', regex=False)
        .str.replace('caer', 'clear', regex=False)
        .str.replace('smokey', 'smoky', regex=False)
        .str.replace('varable', 'variable', regex=False)
    )

    return df

# 3) CORRECTING COORDINATES -----------------------------------------------------------

#Dealing with formatting issues/text inputs
def normalize_coords(df, lat_col='Latitude', lon_col='Longitude', verbose=True, save_fixes_file=None):

    missing_vals_upper = {
        'N/A', 'NA', '', 'NULL', 'NONE', '--', '-', 'NOT GIVEN', 'NN/A'
        'NO OB', 'OBSERVED BUT TOO FADED TO READ', 'DOES NOOT SAY',
        'BY LAND', 'UNREADABLE', 'UNCOMPREHENSBILE', 'NON GIVEN',
        'DOES NOT SAY', 'DOES NTO SAY W', 'NO OBSERVATION', 'NOT SAY',
        'JOT GIVEN', 'NO OBSERVATION.', 'NO OB.', 'DOES NOT SAY W',
        'NAN', 'GIVEN', 'MERIDIAN', '?'
        }

    def norm_coord(val):
        if not isinstance(val, str):
            return val
        s = val.strip().upper()
        if s in missing_vals_upper:
            return np.nan
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'(\d)\s*([NSWE])$', r'\1 \2', s)
        s = re.sub(r'^(\d{2,3})(\d{2}) ([NSWE])$', r'\1 \2 \3', s)
        s = re.sub(r'(\d+)\s+(\d+)\s+([NSWE])$', r'\1 \2 \3', s)
        return s.strip()

    def safe_diff(a, b):
        return ~a.fillna('__NA__').eq(b.fillna('__NA__'))

    all_unique_fixes = []

    for col in [lat_col, lon_col]:
        before = df[col].copy()
        df[col] = df[col].apply(norm_coord)
        after = df[col]
        mask = safe_diff(before, after)

        if verbose:
            changed = df.loc[mask, ['LogBook ID', 'ID', col]]
            if not changed.empty:
                print(f"\n{col} entries corrected ({len(changed)}):")
                print(changed)

        # Collect unique fixes
        unique_corrections = pd.DataFrame({
            'Column': col,
            'Before': before[mask],
            'After': after[mask]
        }).drop_duplicates().reset_index(drop=True)

        all_unique_fixes.append(unique_corrections)

    if save_fixes_file:
        # Combine both columns' fixes and save to a txt file (tab separated)
        fixes_df = pd.concat(all_unique_fixes, ignore_index=True)
        fixes_df.to_csv(save_fixes_file, sep='\t', index=False)
        if verbose:
            print(f"\nAll unique fixes saved to: {save_fixes_file}")

    return df

def convert_miles_to_dms(value, col=None):
    value_str = str(value).strip()
    
    # Special case: if latitude contains "equator", replace with "00 00 N"
    if col == 'Latitude' and 'equator' in value_str.lower():
        print(f"Original: {value_str} -> Converted: 00 00 N")
        return "00 00 N"
    
    # Match "20 miles N", "10 mi E", "M" as shorthand, etc.
    match = re.search(r'(\d+)\s*(miles?|mi|M)\s*([NSWE])', value_str, re.IGNORECASE)
    if match:
        miles = float(match.group(1))
        direction = match.group(3).upper()
        degmin = (miles / 69) * 60
        deg = int(degmin // 60)
        minutes = round(degmin % 60)
        converted = f"{deg:02} {minutes:02} {direction}"
        print(f"Original: {value_str} -> Converted: {converted}")
        return converted
    
    return value_str

def flag_and_convert_miles(df):
    for col in ['Latitude', 'Longitude']:
        mask = df[col].fillna('').str.contains(r'\d+\s*(?:miles?|mi|M)\s*[NSWE]', case=False)
        
        # Use lambda to pass col name to convert_miles_to_dms
        df[col] = df[col].apply(lambda v: convert_miles_to_dms(v, col=col))
    return df
    
#Check for lingering digit issues
def flag_coords_too_many_digits(df, lat_col='Latitude', lon_col='Longitude', verbose=True):
    """
    Flags coordinates where the second number has 3 or more digits.
    Prints a single message if none are found.
    """
    pat = r'^\d{1,3}\s\d{3,}\s[NSWE]$'
    mask_lon = df[lon_col].notna() & df[lon_col].str.match(pat)
    flagged_lon = df.loc[mask_lon, ['ID', 'LogBook ID', 'Entry Date', lon_col]]
    mask_lat = df[lat_col].notna() & df[lat_col].str.match(pat)
    flagged_lat = df.loc[mask_lat, ['ID', 'LogBook ID', 'Entry Date', lat_col]]

    if verbose:
        if flagged_lon.empty and flagged_lat.empty:
            print("No coordinate values found with 3+ digits in the second number.")
        else:
            if not flagged_lon.empty:
                print(f"Longitude values with 3+ digits in the second number: {len(flagged_lon)}")
                print(flagged_lon)
            if not flagged_lat.empty:
                print(f"Latitude values with 3+ digits in the second number: {len(flagged_lat)}")
                print(flagged_lat)

    return flagged_lon.index.tolist(), flagged_lat.index.tolist()


# def correct_coord(df, row_idx, col, log_path="coord_direction_corrections_log.txt", *, force_both=False):
#     """
#     Show consistent context (ID, LogBook ID, Entry Date, Latitude, Longitude),
#     prompt for a correction, and log the change.

#     Inputs:
#       - Press Enter: skip
#       - 'nan'       : set missing (np.nan)
#       - 'both'      : interactively enter new values for BOTH Latitude and Longitude (no inference)
#     """
#     # Context block (unchanged)
#     context_cols = [c for c in ['ID', 'LogBook ID', 'Entry Date', 'Latitude', 'Longitude'] if c in df.columns]
#     start = max(0, row_idx - 5)
#     end   = min(len(df), row_idx + 6)
#     print(f"\nContext for {col} needing correction (row {row_idx}, ID {df.at[row_idx, 'ID']}):\n")
#     print(df.loc[start:end, context_cols].to_string(index=False))

#     # Current value for the target column
#     old_val = df.at[row_idx, col]
#     print(f"\nCurrent value: {old_val}")

#     # NEW: allow the caller to force the 'both' path
#     if force_both:
#         user = 'both'
#     else:
#         user = input(f"Enter corrected value for {col} (Enter=skip, 'nan'=missing, 'both'=edit both): ").strip()

#     if not user and not force_both:
#         print("No correction applied.")
#         return df

#     def _coerce(v):
#         v = v.strip()
#         return np.nan if v.lower() == 'nan' else v

#     # --- BOTH edit path (either user typed 'both' or caller forced it) ---
#     if force_both or user.lower() == 'both':
#         cur_lat = df.at[row_idx, 'Latitude']  if 'Latitude'  in df.columns else None
#         cur_lon = df.at[row_idx, 'Longitude'] if 'Longitude' in df.columns else None
#         print(f"\nEditing BOTH coordinates for row {row_idx} (ID {df.at[row_idx, 'ID']}):")
#         print(f"Current Latitude : {cur_lat}")
#         new_lat_in = input("New Latitude  (Enter=keep current, 'nan'=missing): ").strip()
#         print(f"Current Longitude: {cur_lon}")
#         new_lon_in = input("New Longitude (Enter=keep current, 'nan'=missing): ").strip()

#         if 'Latitude' in df.columns and new_lat_in:
#             old_lat = df.at[row_idx, 'Latitude']
#             df.at[row_idx, 'Latitude'] = _coerce(new_lat_in)
#             if log_path:
#                 with open(log_path, "a", encoding="utf-8") as f:
#                     f.write(f"{datetime.datetime.now()} | ID {df.at[row_idx,'ID']} | "
#                             f"LogBook ID {df.at[row_idx,'LogBook ID']} | Latitude: "
#                             f"'{old_lat}' → '{df.at[row_idx, 'Latitude']}'\n")

#         if 'Longitude' in df.columns and new_lon_in:
#             old_lon = df.at[row_idx, 'Longitude']
#             df.at[row_idx, 'Longitude'] = _coerce(new_lon_in)
#             if log_path:
#                 with open(log_path, "a", encoding="utf-8") as f:
#                     f.write(f"{datetime.datetime.now()} | ID {df.at[row_idx,'ID']} | "
#                             f"LogBook ID {df.at[row_idx,'LogBook ID']} | Longitude: "
#                             f"'{old_lon}' → '{df.at[row_idx, 'Longitude']}'\n")

#         print("Applied updates for both coordinates (where provided).")
#         return df

#     # --- Single-column edit path (unchanged) ---
#     new_val = _coerce(user)
#     if (isinstance(new_val, float) and np.isnan(new_val)) or (str(new_val) != str(old_val)):
#         df.at[row_idx, col] = new_val
#         if log_path:
#             with open(log_path, "a", encoding="utf-8") as f:
#                 f.write(f"{datetime.datetime.now()} | ID {df.at[row_idx,'ID']} | "
#                         f"LogBook ID {df.at[row_idx,'LogBook ID']} | {col}: "
#                         f"'{old_val}' → '{df.at[row_idx, col]}'\n")
#         print("Correction applied and logged.")
#     else:
#         print("No change.")
#     return df

def correct_coord(df, row_idx, col, log_path="coord_direction_corrections_log.txt", *, force_both=False):
    """
    Show context, prompt for correction, and log changes.
    Inputs: Enter=skip, 'nan'=missing, 'both'=edit both coordinates
    """
    
    def _log_change(col_name, old, new):
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} | ID {df.at[row_idx,'ID']} | "
                        f"LogBook ID {df.at[row_idx,'LogBook ID']} | {col_name}: "
                        f"'{old}' → '{new}'\n")
    
    def _coerce(v):
        """Convert input to appropriate type (nan or string)."""
        if not v:
            return None
        v = v.strip()
        return np.nan if v.lower() == 'nan' else v
    
    # Show context
    context_cols = [c for c in ['ID', 'LogBook ID', 'Entry Date', 'Latitude', 'Longitude'] if c in df.columns]
    start, end = max(0, row_idx - 5), min(len(df), row_idx + 6)
    print(f"\nContext for {col} (row {row_idx}, ID {df.at[row_idx, 'ID']}):\n")
    print(df.loc[start:end, context_cols].to_string(index=False))
    print(f"\nCurrent value: {df.at[row_idx, col]}")
    
    # Get user input
    if not force_both:
        user = input(f"Enter corrected {col} (Enter=skip, 'nan'=missing, 'both'=edit both): ").strip()
        if not user:
            print("No correction applied.")
            return df
    else:
        user = 'both'
    
    # Handle 'both' mode
    if user.lower() == 'both':
        print(f"\nEditing BOTH coordinates for row {row_idx}:")
        for coord_col in ['Latitude', 'Longitude']:
            if coord_col in df.columns:
                cur_val = df.at[row_idx, coord_col]
                print(f"Current {coord_col}: {cur_val}")
                new_val = input(f"New {coord_col} (Enter=keep, 'nan'=missing): ").strip()
                if new_val:
                    old_val = df.at[row_idx, coord_col]
                    df.at[row_idx, coord_col] = _coerce(new_val)
                    _log_change(coord_col, old_val, df.at[row_idx, coord_col])
        print("Applied updates.")
        return df
    
    # Handle single column edit
    new_val = _coerce(user)
    old_val = df.at[row_idx, col]
    
    if new_val is not None and ((isinstance(new_val, float) and np.isnan(new_val)) or (str(new_val) != str(old_val))):
        df.at[row_idx, col] = new_val
        _log_change(col, old_val, new_val)
        print("Correction applied and logged.")
    else:
        print("No change.")
    
    return df



# coordinate issue flaggers
def flag_coords_missing_direction(df: pd.DataFrame, lat_col='Latitude', lon_col='Longitude'):
    """
    Find coords missing N/S/E/W suffix.
    Allow 1–3 digits for degrees so 3-digit mistakes get flagged too.
    """
    lon_pattern = r'^\d{1,3}\s\d{1,2}$'  # no E/W present
    lat_pattern = r'^\d{1,3}\s\d{1,2}$'  # no N/S present

    mask_lon = df[lon_col].notna() & df[lon_col].astype(str).str.match(lon_pattern, na=False)
    mask_lat = df[lat_col].notna() & df[lat_col].astype(str).str.match(lat_pattern, na=False)
    return list(df[mask_lon].index), list(df[mask_lat].index)


def flag_direction_symbol_errors(df: pd.DataFrame, lat_col='Latitude', lon_col='Longitude'):
    """
    Longitude incorrectly labeled with N/S; Latitude incorrectly labeled with E/W.
    Allow optional minutes.
    """
    lon_mask = df[lon_col].notna() & df[lon_col].astype(str).str.match(r'^\d{1,3}(?:\s\d{1,2})?\s[NS]$', na=False)
    lat_mask = df[lat_col].notna() & df[lat_col].astype(str).str.match(r'^\d{1,3}(?:\s\d{1,2})?\s[EW]$', na=False)
    return list(df[lon_mask].index), list(df[lat_mask].index)

# flag >180 (lon) and >90 (lat)
def flag_coords_beyond_bounds(df: pd.DataFrame, lat_col='Latitude', lon_col='Longitude'):
    """
    Parse D M Dir and flag lon>180 or lat>90 (or exactly 180/90 with nonzero minutes).
    """
    lon_pat = r'^\s*(\d{1,3})\s+(\d{1,2})\s+([EW])\s*$'
    lat_pat = r'^\s*(\d{1,3})\s+(\d{1,2})\s+([NS])\s*$'

    lon_parts = df[lon_col].astype(str).str.extract(lon_pat, flags=re.I)
    df['_lon_deg'] = pd.to_numeric(lon_parts[0], errors='coerce')
    df['_lon_min'] = pd.to_numeric(lon_parts[1], errors='coerce')

    lat_parts = df[lat_col].astype(str).str.extract(lat_pat, flags=re.I)
    df['_lat_deg'] = pd.to_numeric(lat_parts[0], errors='coerce')
    df['_lat_min'] = pd.to_numeric(lat_parts[1], errors='coerce')

    greater_than_180_mask_lon = ((df['_lon_deg'] > 180) | ((df['_lon_deg'] == 180) & (df['_lon_min'] > 0)))
    greater_than_90_mask_lat  = ((df['_lat_deg'] > 90)  | ((df['_lat_deg'] == 90)  & (df['_lat_min'] > 0)))

    flagged_ids_lon = sorted(set(df.loc[greater_than_180_mask_lon, 'ID']))
    flagged_ids_lat = sorted(set(df.loc[greater_than_90_mask_lat,  'ID']))

    print(f"\nEntries with Longitude outside bounds: {len(flagged_ids_lon)}")
    if flagged_ids_lon:
        print(df.loc[df['ID'].isin(flagged_ids_lon), ['LogBook ID', 'ID', lon_col]])

    print(f"\nEntries with Latitude outside bounds: {len(flagged_ids_lat)}")
    if flagged_ids_lat:
        print(df.loc[df['ID'].isin(flagged_ids_lat), ['LogBook ID', 'ID', lat_col]])

    # Clean helpers
    df.drop(columns=['_lon_deg','_lon_min','_lat_deg','_lat_min'], inplace=True, errors='ignore')
    return flagged_ids_lon, flagged_ids_lat


# batch correction driver
def batch_correct_coords(
    df: pd.DataFrame,
    flag_func,
    lat_col='Latitude',
    lon_col='Longitude',
    log_path: str | None = None
) -> pd.DataFrame:
    """
    Run a flagger that returns (idxs_lon, idxs_lat); interactively correct each.
    """
    idxs_lon, idxs_lat = flag_func(df, lat_col=lat_col, lon_col=lon_col)

    for idx in idxs_lon:
        correct_coord(df, idx, lon_col, log_path=log_path)
    for idx in idxs_lat:
        correct_coord(df, idx, lat_col, log_path=log_path)
    return df


# programmatic correction application by ID 
def apply_coord_corrections(df: pd.DataFrame, corrections: dict, id_col="ID") -> pd.DataFrame:
    """
    Overwrite coords from {ID: {"Longitude": "...", "Latitude": "..."}}.
    """
    for _id, coords in corrections.items():
        mask = df[id_col] == _id
        if not mask.any():
            continue
        for col, val in coords.items():
            df.loc[mask, col] = val
    return df


# review wrapper for outliers (uses the single corrector)
def examine_and_correct_outliers(df: pd.DataFrame, flagged_ids: list, col: str, log_path="coord_direction_corrections_log.txt") -> pd.DataFrame:
    for row_idx in df.index[df['ID'].isin(flagged_ids)]:
        correct_coord(df, row_idx, col, log_path=log_path)
    return df


# save lingering invalid coordinates --- use to review issues left at the end of data cleaning
def save_invalid_coords(df: pd.DataFrame, lat_col='Latitude', lon_col='Longitude', dir_path: str | None = None):
    """
    Write out remaining invalid tokens to txt files for audit. Excludes blanks.
    """
    if dir_path is None:
        dir_path = os.getcwd()

    # Longitude (allow D M or D M S forms ending E/W)
    lon_pat = r'^\d{1,3}(?:\s\d{1,2}){1,2}\s[WE]$'
    lon_data = df[['LogBook ID', 'ID', lat_col, lon_col]].dropna(subset=[lon_col])
    invalid_lon = lon_data[~lon_data[lon_col].astype(str).str.match(lon_pat) & (lon_data[lon_col].astype(str).str.strip() != '')]
    invalid_lon = invalid_lon.sort_values('LogBook ID')
    out_lon = os.path.join(dir_path, 'invalid_longitude_terms.txt')
    with open(out_lon, 'w', encoding="utf-8") as f:
        for _, row in invalid_lon.iterrows():
            f.write(f"LogBook ID: {row['LogBook ID']}, ID: {row['ID']}, Latitude: {row[lat_col]}, Longitude: {row[lon_col]}\n")
    print(f"Invalid Longitude terms (excluding blanks) saved to {out_lon}")

    # Latitude (D M or D M S ending N/S) – use 1–2 deg for "valid" format
    lat_pat = r'^\d{1,2}(?:\s\d{1,2}){1,2}\s[NS]$'
    lat_data = df[['LogBook ID', 'ID', lat_col, lon_col]].dropna(subset=[lat_col])
    invalid_lat = lat_data[~lat_data[lat_col].astype(str).str.match(lat_pat) & (lat_data[lat_col].astype(str).str.strip() != '')]
    invalid_lat = invalid_lat.sort_values('LogBook ID')
    out_lat = os.path.join(dir_path, 'invalid_latitude_terms.txt')
    with open(out_lat, 'w', encoding="utf-8") as f:
        for _, row in invalid_lat.iterrows():
            f.write(f"LogBook ID: {row['LogBook ID']}, ID: {row['ID']}, Latitude: {row[lat_col]}, Longitude: {row[lon_col]}\n")
    print(f"Invalid Latitude terms (excluding blanks) saved to {out_lat}")

    return invalid_lon, invalid_lat

# address issues that may have persisted through main cleaning functions and enforce use of np.nan
def final_coord_cleanup(df: pd.DataFrame, lat_col='Latitude', lon_col='Longitude') -> pd.DataFrame:
    """
    Final pass to fix remaining coordinate text issues.

    Additions:
      - Set records with no trailing direction (N/E/S/W) to NaN (e.g., '44', '71', '00 3? ?').
      - Convert minutes with a single '?' to '0' at that position (e.g., '33 4? S' -> '33 40 S').
      - Treat decimals like '59.20 W' as '59 20 W'.
      - Strip fractional minutes markers like '1/2' (e.g., '104 04 1/2 W' -> '104 04 W').
      - Keep prior rules: remove DR, '??' -> '00', degrees with '?' -> NaN, only cardinal -> NaN, strip stray ']' and '`'.
    """
    def _normalize(val: object, is_lat: bool) -> object:
        if pd.isna(val):
            return val
        s = str(val).strip()
        if s == '':
            return np.nan

        # Light noise removal
        s = s.replace('`', ' ').replace(']', ' ').replace(',', ' ')
        s = re.sub(r'\s+', ' ', s).strip()

        U = s.upper()
        if U in {'NAC', 'NO OB', 'NO OBS', 'NA', 'N/A'} or 'N/A' in U:
            return np.nan

        # Only a single cardinal -> NaN
        if re.fullmatch(r'[NSEW]', U):
            return np.nan

        # Remove DR markers (DR / D R)
        s = re.sub(r'\bD\s*R\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s+', ' ', s).strip()

        # Strip fractional minutes like "1/2" (keep the integer minutes)
        s = re.sub(r'\b1/2\b', '', s)
        s = re.sub(r'\s+', ' ', s).strip()

        # Decimal minute style: "DD.MM Dir" (e.g., 59.20 W -> 59 20 W)
        m_dec = re.fullmatch(r'(?i)\s*(\d{1,3})\.(\d{1,2})\s*([NSEW])\s*', s)
        if m_dec:
            deg, mins, direc = m_dec.groups()
            return f"{int(deg)} {mins.zfill(2)} {direc.upper()}"

        # If there's NO trailing direction at all, we cannot safely normalize -> NaN
        if not re.search(r'(?i)[NSEW]\s*$', s):
            return np.nan

        # Now we know there *is* a trailing direction — normalize "<deg> [<min>] Dir"
        # Allow '?' in minutes, but NOT in degrees.
        m = re.fullmatch(r'(?i)\s*([0-9?]{1,3})(?:\s+([0-9?]{1,2}))?\s+([NSEW])\s*', s)
        if not m:
            # Try a variant that accidentally has extra tokens before direction (we already removed 1/2)
            # If still not matching, give up (leave as-is).
            return s

        deg, mins, direc = m.group(1), m.group(2), m.group(3).upper()

        # Degree with any '?' -> NaN (cannot infer)
        if '?' in deg:
            return np.nan

        # Minutes handling:
        if mins is None:
            mins = '00'
        else:
            # '??' -> '00'
            if mins == '??':
                mins = '00'
            # Single '?' inside minutes -> replace '?' with '0' (e.g., '4?' -> '40', '?5' -> '05')
            elif '?' in mins:
                mins = mins.replace('?', '0')
            # Ensure 2-digit minutes
            mins = mins.zfill(2)

        # Rebuild normalized "D M Dir"
        return f"{int(deg)} {mins} {direc}"

    df[lat_col] = df[lat_col].apply(lambda v: _normalize(v, is_lat=True))
    df[lon_col] = df[lon_col].apply(lambda v: _normalize(v, is_lat=False))
    return df

# 4) PLOT GENERATION FOR DATA VALIDATION -------------------------------------------------------------------

# convert dms coords to decimal coords
def dms_to_decimal(coord: object) -> float:
    """
    Convert strings like '38 08 N' or '70 15 30 W' to decimal degrees.
    Returns np.nan for non-strings, NaNs, or non-matching formats.
    """
    if pd.isna(coord) or not isinstance(coord, str):
        return np.nan

    s = coord.strip().upper()
    # degrees minutes [optional seconds] + direction
    m = re.match(r'^(\d{1,3})\s+(\d{1,2})(?:\s+(\d{1,2}))?\s*([NSEW])$', s)
    if not m:
        return np.nan

    deg, minute, second, hemi = m.groups()
    sec = float(second) if second is not None else 0.0

    dec = float(deg) + float(minute)/60.0 + sec/3600.0
    if hemi in ('S', 'W'):
        dec = -dec
    return dec

#add decimal values to dataframe
def add_decimal_columns(
    df: pd.DataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    out_lat: str = 'Latitude_decimal',
    out_lon: str = 'Longitude_decimal'
) -> pd.DataFrame:
    """
    Compute decimal-degree columns from DMS strings and insert each new column
    right after its source column. Invalid/missing inputs -> np.nan.
    Returns the a modified copy of the input DataFrame.
    """
    df = df.copy()

    # compute decimals
    lat_decimal = df[lat_col].apply(dms_to_decimal)
    lon_decimal = df[lon_col].apply(dms_to_decimal)

    # insert after original columns
    lat_idx = df.columns.get_loc(lat_col)
    lon_idx = df.columns.get_loc(lon_col)
    # if lon is after lat, inserting lat first keeps lon index valid; if reversed, swap order
    if lon_idx > lat_idx:
        df.insert(lat_idx + 1, out_lat, lat_decimal)
        df.insert(df.columns.get_loc(lon_col) + 1, out_lon, lon_decimal)
    else:
        df.insert(lon_idx + 1, out_lon, lon_decimal)
        df.insert(df.columns.get_loc(lat_col) + 1, out_lat, lat_decimal)

    return df

#flag potentially problematic jumps in coordinates
def flag_unrealistic_coord_jumps(
    df: pd.DataFrame,
    time_col: str = 'Entry Date Time',
    logbook_col: str = 'LogBook ID',
    lat_col: str = 'Latitude_decimal',
    lon_col: str = 'Longitude_decimal',
    time_format: str | None = '%Y-%m-%d %H:%M:%S',
    time_delta_seconds: int = 60*60*24*2,  # 2 days
    latlon_delta_deg: float = 10.0,        # flag if |Δlat|>10 OR |Δlon|>10
    lon_delta_upper_limit: float = 320.0   # ignore wrap if absurd jump
) -> pd.DataFrame:
    """
    Flags rows where the coordinate jump (vs previous entry in the SAME logbook)
    is large within a short time window.

    Adds columns:
      - prev_time, delta_time_s
      - dlat_deg, dlon_deg
      - coord_diff (boolean flag)
    """
    df = df.copy()

    # robust datetime parse
    if time_format:
        df[time_col] = pd.to_datetime(df[time_col], format=time_format, errors='coerce')
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    # work within each logbook and sort by time
    df.sort_values([logbook_col, time_col], inplace=True)

    # previous row within the same logbook
    df['prev_time'] = df.groupby(logbook_col)[time_col].shift()
    df['prev_lat']  = df.groupby(logbook_col)[lat_col].shift()
    df['prev_lon']  = df.groupby(logbook_col)[lon_col].shift()

    # time delta (seconds)
    dt = (df[time_col] - df['prev_time']).dt.total_seconds()
    df['delta_time_s'] = dt

    # angular deltas
    dlat = (df[lat_col] - df['prev_lat']).abs()
    # handle longitude across dateline by taking the shorter wrap-around difference
    raw_dlon = (df[lon_col] - df['prev_lon']).abs()
    wrapped_dlon = 360 - raw_dlon
    dlon = np.where(
        (raw_dlon.notna()) & (wrapped_dlon.notna()),
        np.minimum(raw_dlon, wrapped_dlon),
        np.nan
    )
    df['dlat_deg'] = dlat
    df['dlon_deg'] = dlon

    cond_large_move = ( (dlat > latlon_delta_deg) | (dlon > latlon_delta_deg) )
    cond_time_close = (dt.notna()) & (dt <= time_delta_seconds)
    cond_same_book  = True  
    cond_lon_upper  = (raw_dlon < lon_delta_upper_limit) | raw_dlon.isna()

    df['coord_diff'] = cond_large_move & cond_time_close & cond_lon_upper

    return df

    return decimal

# wrapper function to only plot specified combinations of new entries
def plot_new_entries(
    df_corrected: pd.DataFrame,
    new_rows: pd.DataFrame,
    figures_dir: str | None = None,
    export_label: str = "latest export",
    plot_scope: str = "new_only",           # "new_only" or "all_from_new_logbooks"
    exclude_logbooks: list[str] | None = None,
    ncols: int = 2,
    projection = ccrs.Robinson(),
    point_marker: str = "+",
    dpi: int = 300,
    save: bool = False,
    
):
    """
    A thin wrapper around `plot_logbook` that lays out multiple logbooks in a grid.

    - new_only: plots only new (LogBook ID, Entry Date) points for each affected logbook
    - all_from_new_logbooks: plots ALL entries for any logbook that has >=1 new point
    """
    exclude_logbooks = set(exclude_logbooks or [])
    new_rows = new_rows.copy()
    df = df_corrected.copy()

    # Normalize times for keying
    new_rows['Entry Date'] = pd.to_datetime(new_rows['Entry Date'], errors='coerce').dt.floor('min')
    df['Entry Date'] = pd.to_datetime(df['Entry Date'], errors='coerce', format='mixed').dt.normalize()

    # Which logbooks are in-scope?
    logbooks_with_new = (
        new_rows['LogBook ID'].dropna().astype(str).unique().tolist()
    )
    logbooks_with_new = [lb for lb in logbooks_with_new if lb not in exclude_logbooks]

    if not logbooks_with_new:
        print("Nothing to plot: no logbooks with new entries (after exclusions).")
        return None, [], {'n_logbooks': 0, 'total_points': 0, 'logbooks': []}

    # Filter df rows to those logbooks and require valid coords
    if plot_scope == "new_only":
        base = df[df['LogBook ID'].isin(logbooks_with_new)].copy()
        # keep all rows for these logbooks, but plot_logbook will filter with only_new=True
        title_prefix = f"All new entries from {export_label}"
        only_new_flag = True
    elif plot_scope == "all_from_new_logbooks":
        base = df[df['LogBook ID'].isin(logbooks_with_new)].copy()
        title_prefix = "Complete logbooks with new entries"
        only_new_flag = False
    else:
        raise ValueError("plot_scope must be 'new_only' or 'all_from_new_logbooks'")

    base = base[base['Latitude_decimal'].notna() & base['Longitude_decimal'].notna()].copy()
    if base.empty:
        print("Nothing to plot (valid coordinates not found in selected scope).")
        return None, [], {'n_logbooks': 0, 'total_points': 0, 'logbooks': []}

    # Lay out a grid and delegate the actual drawing to plot_logbook
    logbooks = base['LogBook ID'].dropna().unique().tolist()
    n_logbooks = len(logbooks)
    nrows = math.ceil(n_logbooks / ncols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(14, 6 * nrows),
        subplot_kw={'projection': projection},
    )
    plt.subplots_adjust(hspace=0.02, wspace = 0.5)  
    axes = (axes.ravel().tolist() if hasattr(axes, "ravel") else [axes])

    total_points_all = 0
    for i, logbook in enumerate(logbooks):
        ax = axes[i]
        # Delegate: draw into this axes
        _, _, npts = plot_logbook(
            base, logbook,
            only_new=only_new_flag,
            new_rows=new_rows if only_new_flag else None,
            projection=projection,
            marker=point_marker,
            title_extra=title_prefix,
            ax=ax,           
            show=False,       
            save=save        
        )
        total_points_all += npts

    # remove any spare axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle(
        f"{title_prefix}\n(new logbooks and logbooks with entries made before and after {export_label})",
        fontsize=12
    )
    plt.tight_layout()

    if figures_dir:
        os.makedirs(figures_dir, exist_ok=True)
        fname = 'all_new_cleaned_entries.png' if plot_scope == "new_only" else 'complete_logbooks_w_new_cleaned_entries.png'
        out_path = os.path.join(figures_dir, fname)
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)
        print(f"Saved figure: {out_path}")

    plt.show()

    return fig, axes, {
        'n_logbooks': n_logbooks,
        'total_points': total_points_all,
        'logbooks': logbooks,
        'scope': plot_scope
    }

def plot_logbook(
    df: pd.DataFrame,
    logbook_id: str,
    years=None,
    year_range=None,
    only_new: bool = False,
    new_rows: pd.DataFrame | None = None,
    annotate: bool = False,
    annotate_field: str = "ID",
    annotate_max: int = 200,
    projection = ccrs.Robinson(),
    marker: str = "+",
    figures_dir: str | None = None,
    filename: str | None = None,
    title_extra: str | None = None,
    dpi: int = 300,
    highlight_idx: int | None = None,
    prev_idx: int | None = None,
    next_idx: int | None = None,
    highlight_marker: str = "*",
    highlight_size: int = 90,
    neighbor_marker: str = "o",
    neighbor_size: int = 70,
    *,
    ax=None,                # NEW: draw into this axes if provided
    show: bool = True,      # NEW: control plt.show()
    save: bool = True       # NEW: control saving from inside this function
):
    if years is not None and year_range is not None:
        raise ValueError("Use either `years` or `year_range`, not both.")

    df = df.copy()
    if 'Entry Date Time' in df.columns:
        df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')
    else:
        df['Entry Date Time'] = pd.to_datetime(df['Entry Date'], errors='coerce')

    df_log = df[
        (df['LogBook ID'] == logbook_id)
        & df['Latitude_decimal'].notna()
        & df['Longitude_decimal'].notna()
    ].copy()
    if df_log.empty:
        print(f"No valid coordinates to plot for: {logbook_id}")
        return None, None, 0

    if years is not None:
        if isinstance(years, int):
            years = [years]
        df_log = df_log[df_log['Entry Date Time'].dt.year.isin(years)]
    elif year_range is not None:
        y0, y1 = year_range
        df_log = df_log[(df_log['Entry Date Time'].dt.year >= y0) &
                        (df_log['Entry Date Time'].dt.year <= y1)]
    if df_log.empty:
        print(f"No rows remain after year filtering for: {logbook_id}")
        return None, None, 0

    if only_new:
        if new_rows is None:
            raise ValueError("only_new=True requires `new_rows` DataFrame.")
        nr = new_rows.copy()
        nr['Entry Date'] = pd.to_datetime(nr['Entry Date'], errors='coerce').dt.floor('min')
        if 'Entry Date' in df_log.columns:
            df_log['Entry Date'] = pd.to_datetime(df_log['Entry Date'], errors='coerce').dt.normalize()
        else:
            df_log['Entry Date'] = df_log['Entry Date Time'].dt.normalize()
        keys_new = set(zip(nr['LogBook ID'], nr['Entry Date']))
        df_log = df_log[[(lb, dt) in keys_new for lb, dt in zip(df_log['LogBook ID'], df_log['Entry Date'])]]
        if df_log.empty:
            print(f"No NEW rows to plot for: {logbook_id}")
            return None, None, 0

    # --- figure/axes handling ---
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': projection})
        created_fig = True
    else:
        fig = ax.figure
        # ensure the target axes has a projection
        try:
            _ = ax.projection
        except Exception:
            raise ValueError("Provided `ax` must be a Cartopy GeoAxes with a projection.")

    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#b0b0b0', edgecolor='none')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    years_present = sorted(df_log['Entry Date Time'].dt.year.dropna().unique())
    total_points = 0
    for yr in years_present:
        df_y = df_log[df_log['Entry Date Time'].dt.year == yr]
        ax.plot(
            df_y['Longitude_decimal'], df_y['Latitude_decimal'],
            marker,
            transform=ccrs.PlateCarree(),
            label=str(yr)
        )
        total_points += len(df_y)

    transform = ccrs.PlateCarree()
    def _scatter_idx(idx, label, *, size, marker, hollow=False, z=5, **extra):
        if idx is None or idx not in df_log.index:
            return
        r = df_log.loc[idx]
        lat = r.get('Latitude_decimal'); lon = r.get('Longitude_decimal')
        if pd.isna(lat) or pd.isna(lon): return
        kw = dict(s=size, marker=marker, zorder=z, label=label, transform=transform)
        if hollow: kw.update(facecolors='none', edgecolors='k', linewidths=1.2)
        ax.scatter([lon], [lat], **kw)

    _scatter_idx(prev_idx, "Previous", size=neighbor_size, marker=neighbor_marker, hollow=True, z=5)
    _scatter_idx(next_idx, "Next",     size=neighbor_size, marker=neighbor_marker, hollow=True, z=5)
    _scatter_idx(highlight_idx, "Inspecting", size=highlight_size, marker=highlight_marker, hollow=False, z=6, color='k')

    if annotate and total_points <= annotate_max and annotate_field in df_log.columns:
        for _, row in df_log.iterrows():
            ax.text(
                row['Longitude_decimal'], row['Latitude_decimal'],
                str(row[annotate_field]),
                transform=transform,
                fontsize=2, ha='right', va='bottom'
            )

    researcher = df_log['Researcher'].iloc[0] if 'Researcher' in df_log.columns and len(df_log) else 'Unknown'
    base_title = f"{logbook_id}\n{researcher} | {total_points} entrie{'s' if total_points!=1 else ''}"
    if title_extra: base_title += f" — {title_extra}"
    ax.set_title(base_title, fontsize=12)

    if len(years_present) >= 2:
        ax.legend(loc='lower left', fontsize=8)

    # Save/show only if we created the figure (avoid duplicates when used as a panel)
    if created_fig and figures_dir and save:
        os.makedirs(figures_dir, exist_ok=True)
        if not filename:
            safe = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in logbook_id)
            filename = f"{safe}__single_logbook.png"
        out_path = os.path.join(figures_dir, filename)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {out_path}")

    if created_fig and show:
        plt.show()

    return fig, ax, total_points


def _filter_logbook_rows( df, logbook_id, years=None, year_range=None, *, only_new=False, new_rows=None):
    """Mirror the filtering in plot_logbook for consistent subsets."""
    df = df.copy()

    # ensure datetime column exists
    if 'Entry Date Time' in df.columns:
        df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')
    else:
        df['Entry Date Time'] = pd.to_datetime(df['Entry Date'], errors='coerce')

    # base filter: chosen logbook + valid coords
    q = (
        (df['LogBook ID'] == logbook_id)
        & df['Latitude_decimal'].notna()
        & df['Longitude_decimal'].notna()
    )
    df_log = df.loc[q].copy()

    # year filters
    if years is not None:
        if isinstance(years, int):
            years = [years]
        df_log = df_log[df_log['Entry Date Time'].dt.year.isin(years)]
    elif year_range is not None:
        y0, y1 = year_range
        df_log = df_log[
            (df_log['Entry Date Time'].dt.year >= y0) &
            (df_log['Entry Date Time'].dt.year <= y1)
        ]

    if only_new:
        if new_rows is None:
            raise ValueError("only_new=True requires `new_rows`.")
        nr = new_rows.copy()
        nr['Entry Date'] = pd.to_datetime(nr['Entry Date'], errors='coerce').dt.normalize()
        if 'Entry Date' in df.columns:
            df['Entry Date'] = pd.to_datetime(df['Entry Date'], errors='coerce').dt.normalize()
        else:
            df['Entry Date'] = pd.to_datetime(df['Entry Date Time'], errors='coerce').dt.normalize()
        keys_new = set(zip(nr['LogBook ID'], nr['Entry Date']))
        df = df[[ (lb, dt) in keys_new for lb, dt in zip(df['LogBook ID'], df['Entry Date']) ]]

    return df_log

def _build_full_context_df(
    df, df_log_idx, current_idx, prev_idx, next_idx,
    flagged_col='coord_diff', include_decimals=True
):
    # --- show only a small window around the current row ---
    WINDOW = 10  # change here if you ever want a different size
    seq = list(df_log_idx)
    if current_idx in seq:
        pos = seq.index(current_idx)
    else:
        pos = len(seq) // 2  # fallback (shouldn't happen)

    half = WINDOW // 2
    lo = max(0, pos - half)
    hi = min(len(seq), lo + WINDOW)
    lo = max(0, hi - WINDOW)  # keep WINDOW rows if we're near the end
    idxs = seq[lo:hi]

    # choose date column
    date_col = 'Entry Date' if 'Entry Date' in df.columns else (
        'Entry Date Time' if 'Entry Date Time' in df.columns else None
    )

    cols = ['ID', 'LogBook ID']
    if date_col: cols.append(date_col)
    cols += ['Latitude', 'Longitude']
    if include_decimals:
        if 'Latitude_decimal' in df.columns:  cols.append('Latitude_decimal')
        if 'Longitude_decimal' in df.columns: cols.append('Longitude_decimal')
    cols = [c for c in cols if c in df.columns]

    # build context table for the window only
    ctx = df.loc[idxs, cols].copy()
    ctx.insert(0, 'IDX', ctx.index)

    # FLAG column if available
    if flagged_col in df.columns:
        ctx['FLAG'] = df.loc[idxs, flagged_col].astype('boolean').fillna(False)

    # MARK prev/current/next if they fall inside the window
    ctx.insert(0, 'MARK', '')
    if current_idx in ctx['IDX'].values:
        ctx.loc[ctx['IDX'] == current_idx, 'MARK'] = 'CURRENT'
    if prev_idx is not None and prev_idx in ctx['IDX'].values:
        ctx.loc[ctx['IDX'] == prev_idx, 'MARK'] = 'prev'
    if next_idx is not None and next_idx in ctx['IDX'].values:
        ctx.loc[ctx['IDX'] == next_idx, 'MARK'] = 'next'

    return ctx


def _print_full_context_df(ctx):
    # show all rows without truncation
    with pd.option_context('display.max_rows', None, 'display.width', 0):
        print(ctx.to_string(index=False))


# def inspect_and_correct_logbook_flags(
#     df: pd.DataFrame,
#     logbook_id: str,
#     flagged_col: str = 'coord_diff',
#     *,
#     years=None,
#     year_range=None,
#     only_new: bool = False,
#     new_rows: pd.DataFrame | None = None,
#     annotate: bool = False,
#     annotate_field: str = "ID",
#     annotate_max: int = 200,
#     projection=None,
#     marker: str = "+",
#     figures_dir: str | None = None,
#     filename: str | None = None,
#     title_extra: str | None = None,
#     dpi: int = 300,
#     log_path: str = None,
#     recompute_decimal_after_each: bool = True,
#     recompute_decimal_fn=None,     # e.g., upsert_decimal_columns
#     reflag_after_each: bool = True,
#     reflag_fn=None,                # e.g., flag_unrealistic_coord_jumps
# ):
#     if projection is None:
#         projection = ccrs.Robinson() if ccrs is not None else None

#     # Build the ordered list of indices (by time) for this logbook with flags
#     df_log = _filter_logbook_rows(
#         df, logbook_id, years=years, year_range=year_range,
#         only_new=only_new, new_rows=new_rows
#     )

#     if df_log.empty:
#         print(f"No rows to inspect for: {logbook_id}")
#         return df

#     if flagged_col not in df.columns:
#         print(f"Flag column '{flagged_col}' not found.")
#         return df

#     flagged_here = df_log.index[df.loc[df_log.index, flagged_col] == True]
#     if len(flagged_here) == 0:
#         print(f"No flagged rows to inspect for: {logbook_id}")
#         return df

#     # Sort by Entry Date Time for consistent stepping
#     df_log_time = df.loc[flagged_here].copy()
#     if 'Entry Date Time' in df_log_time.columns:
#         df_log_time['Entry Date Time'] = pd.to_datetime(df_log_time['Entry Date Time'], errors='coerce')
#         review_order = df_log_time.sort_values('Entry Date Time', na_position='last').index.tolist()
#     else:
#         review_order = sorted(flagged_here)

#     # Precompute the full, time-ordered list of indices for this logbook
#     df_log_all = df.loc[df_log.index].copy()
#     df_log_all['Entry Date Time'] = pd.to_datetime(df_log_all['Entry Date Time'], errors='coerce')
#     order_all = df_log_all.sort_values('Entry Date Time', na_position='last').index.tolist()

#     print(f"\nReviewing {len(review_order)} flagged entries for: {logbook_id}")
#     print("Commands: [c]urrent, [p]revious, [n]ext, [s]kip, [q]uit")
#     print("After choosing target row, choose column: lat / lon / both (or Enter to skip)\n")

#     for idx in review_order:
#         # neighbors within this logbook’s time-ordered indices
#         pos = order_all.index(idx)
#         prev_idx = order_all[pos-1] if pos > 0 else None
#         next_idx = order_all[pos+1] if pos < len(order_all)-1 else None
    
#         # ---- FULL CONTEXT first ----
#         ctx = _build_full_context_df(
#             df, df_log.index, current_idx=idx, prev_idx=prev_idx, next_idx=next_idx,
#             flagged_col=flagged_col, include_decimals=True
#         )
#         print(f"\nContext for logbook: {logbook_id}  (flagged ID {df.at[idx,'ID']})")
#         _print_full_context_df(ctx)
    
#         # ---- plot current with a single star ----
#         _ = plot_logbook(
#             df=df,
#             logbook_id=logbook_id,
#             years=years,
#             year_range=year_range,
#             only_new=only_new,
#             new_rows=new_rows,
#             annotate=annotate,
#             annotate_field=annotate_field,
#             annotate_max=annotate_max,
#             projection=projection,
#             marker=marker,
#             figures_dir=figures_dir,
#             filename=filename,
#             title_extra=title_extra,
#             dpi=dpi,
#             highlight_idx=idx,
#             prev_idx=prev_idx,
#             next_idx=next_idx,

#         )
    
#         # helper to edit + recompute + reflag
#         def _edit_and_refresh(target_idx: int, col_choice: str):
#             nonlocal df
#             if col_choice == 'lat':
#                 df = correct_coord(df, target_idx, col='Latitude', log_path=log_path)
#             elif col_choice == 'lon':
#                 df = correct_coord(df, target_idx, col='Longitude', log_path=log_path)
#             elif col_choice == 'both':
#                 df = correct_coord(df, target_idx, col='Latitude', log_path=log_path, force_both=True)
#             else:
#                 print("No edit made.")
#                 return
        
#             if recompute_decimal_after_each and callable(recompute_decimal_fn):
#                 df = recompute_decimal_fn(df)
#             if reflag_after_each and callable(reflag_fn):
#                 df = reflag_fn(df)

    
#         # ---- selection prompt ----
#         prompt = (
#             f"[ID {df.at[idx,'ID']}] Edit which row? "
#             "[c]urrent / [p]rev / [n]ext / [s]kip / [q]uit "
#             "or type 'id 32444' / 'idx 32793' (or just a number) to edit a different row: "
#         )
#         choose = input(prompt).strip().lower()
    
#         if choose == 'q':
#             print("Stopping review.")
#             break
#         if choose in ('s', ''):
#             continue
    
#         # parse free-form choices
#         target = None
#         if choose in ('c', 'p', 'n'):
#             if choose == 'p':
#                 if prev_idx is None:
#                     print("No previous row in this logbook.")
#                     continue
#                 target = prev_idx
#             elif choose == 'n':
#                 if next_idx is None:
#                     print("No next row in this logbook.")
#                     continue
#                 target = next_idx
#             else:
#                 target = idx
#         else:
#             # accept: 'id 32444' or 'idx 32793' or just '32444'
#             tokens = choose.split()
#             if len(tokens) == 2 and tokens[0] in ('id', 'idx') and tokens[1].isdigit():
#                 kind, num = tokens[0], int(tokens[1])
#                 if kind == 'id':
#                     matches = df.loc[df_log.index][df.loc[df_log.index, 'ID'] == num]
#                     if matches.empty:
#                         print(f"ID {num} not found in this logbook subset.")
#                         continue
#                     target = matches.index[0]
#                 else:  # 'idx'
#                     if num not in df_log.index:
#                         print(f"Index {num} not in this logbook subset.")
#                         continue
#                     target = num
#             elif choose.isdigit():
#                 num = int(choose)
#                 matches = df.loc[df_log.index][df.loc[df_log.index, 'ID'] == num]
#                 if not matches.empty:
#                     target = matches.index[0]
#                 elif num in df_log.index:
#                     target = num
#                 else:
#                     print(f"'{choose}' not recognized as ID or index in this logbook.")
#                     continue
#             else:
#                 print("Input not recognized. Skipping.")
#                 continue
    
#             # optional: confirm visually by starring the chosen target
#             _ = plot_logbook(
#                 df=df,
#                 logbook_id=logbook_id,
#                 years=years,
#                 year_range=year_range,
#                 only_new=only_new,
#                 new_rows=new_rows,
#                 annotate=annotate,
#                 annotate_field=annotate_field,
#                 annotate_max=annotate_max,
#                 projection=projection,
#                 marker=marker,
#                 figures_dir=figures_dir,
#                 filename=filename,
#                 title_extra=title_extra,
#                 dpi=dpi,
#                 highlight_idx=target,
#             )
    
#         # which column?
#         col_choice = input("Edit which column? [lat] / [lon] / [both] (Enter=skip): ").strip().lower()
#         if not col_choice:
#             print("Skipped.")
#             continue
    
#         _edit_and_refresh(target, col_choice)


#     print("\nDone reviewing flagged entries.")
#     return df

def inspect_and_correct_logbook_flags(
    df: pd.DataFrame,
    logbook_id: str,
    flagged_col: str = 'coord_diff',
    *,
    years=None,
    year_range=None,
    only_new: bool = False,
    new_rows: pd.DataFrame | None = None,
    annotate: bool = False,
    annotate_field: str = "ID",
    annotate_max: int = 200,
    projection=None,
    marker: str = "+",
    figures_dir: str | None = None,
    filename: str | None = None,
    title_extra: str | None = None,
    dpi: int = 300,
    log_path: str = None,
    recompute_decimal_after_each: bool = True,
    recompute_decimal_fn=None,
    reflag_after_each: bool = True,
    reflag_fn=None,
    skip_list_path: str = None,  # NEW: path to skip list file
):
    if projection is None:
        projection = ccrs.Robinson() if ccrs is not None else None

    # NEW: Load skip list
    skip_set = set()
    if skip_list_path and os.path.exists(skip_list_path):
        with open(skip_list_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        skip_set.add(int(line))
                    except ValueError:
                        pass
        if skip_set:
            print(f"Loaded {len(skip_set)} IDs from skip list: {skip_list_path}")

    # NEW: Helper to add to skip list
    def _add_to_skip_list(row_id):
        if skip_list_path:
            with open(skip_list_path, 'a') as f:
                f.write(f"{row_id}\n")
            skip_set.add(row_id)
            print(f"Added ID {row_id} to skip list.")

    # Build the ordered list of indices (by time) for this logbook with flags
    df_log = _filter_logbook_rows(
        df, logbook_id, years=years, year_range=year_range,
        only_new=only_new, new_rows=new_rows
    )

    if df_log.empty:
        print(f"No rows to inspect for: {logbook_id}")
        return df

    if flagged_col not in df.columns:
        print(f"Flag column '{flagged_col}' not found.")
        return df

    flagged_here = df_log.index[df.loc[df_log.index, flagged_col] == True]
    
    # NEW: Filter out IDs in skip list
    if skip_set and 'ID' in df.columns:
        original_count = len(flagged_here)
        flagged_here = [idx for idx in flagged_here if df.at[idx, 'ID'] not in skip_set]
        skipped_count = original_count - len(flagged_here)
        if skipped_count > 0:
            print(f"Filtered out {skipped_count} entries from skip list.")
    
    if len(flagged_here) == 0:
        print(f"No flagged rows to inspect for: {logbook_id}")
        return df

    # Sort by Entry Date Time for consistent stepping
    df_log_time = df.loc[flagged_here].copy()
    if 'Entry Date Time' in df_log_time.columns:
        df_log_time['Entry Date Time'] = pd.to_datetime(df_log_time['Entry Date Time'], errors='coerce')
        review_order = df_log_time.sort_values('Entry Date Time', na_position='last').index.tolist()
    else:
        review_order = sorted(flagged_here)

    # Precompute the full, time-ordered list of indices for this logbook
    df_log_all = df.loc[df_log.index].copy()
    df_log_all['Entry Date Time'] = pd.to_datetime(df_log_all['Entry Date Time'], errors='coerce')
    order_all = df_log_all.sort_values('Entry Date Time', na_position='last').index.tolist()

    print(f"\nReviewing {len(review_order)} flagged entries for: {logbook_id}")
    print("Commands: [c]urrent, [p]revious, [n]ext, [s]kip, [o]k (add to skip list), [q]uit")
    print("After choosing target row, choose column: lat / lon / both (or Enter to skip)\n")

    for idx in review_order:
        # neighbors within this logbook's time-ordered indices
        pos = order_all.index(idx)
        prev_idx = order_all[pos-1] if pos > 0 else None
        next_idx = order_all[pos+1] if pos < len(order_all)-1 else None
    
        # ---- FULL CONTEXT first ----
        ctx = _build_full_context_df(
            df, df_log.index, current_idx=idx, prev_idx=prev_idx, next_idx=next_idx,
            flagged_col=flagged_col, include_decimals=True
        )
        print(f"\nContext for logbook: {logbook_id}  (flagged ID {df.at[idx,'ID']})")
        _print_full_context_df(ctx)
    
        # ---- plot current with a single star ----
        _ = plot_logbook(
            df=df,
            logbook_id=logbook_id,
            years=years,
            year_range=year_range,
            only_new=only_new,
            new_rows=new_rows,
            annotate=annotate,
            annotate_field=annotate_field,
            annotate_max=annotate_max,
            projection=projection,
            marker=marker,
            figures_dir=figures_dir,
            filename=filename,
            title_extra=title_extra,
            dpi=dpi,
            highlight_idx=idx,
            prev_idx=prev_idx,
            next_idx=next_idx,
        )
    
        # helper to edit + recompute + reflag
        def _edit_and_refresh(target_idx: int, col_choice: str):
            nonlocal df
            if col_choice == 'lat':
                df = correct_coord(df, target_idx, col='Latitude', log_path=log_path)
            elif col_choice == 'lon':
                df = correct_coord(df, target_idx, col='Longitude', log_path=log_path)
            elif col_choice == 'both':
                df = correct_coord(df, target_idx, col='Latitude', log_path=log_path, force_both=True)
            else:
                print("No edit made.")
                return
        
            if recompute_decimal_after_each and callable(recompute_decimal_fn):
                df = recompute_decimal_fn(df)
            if reflag_after_each and callable(reflag_fn):
                df = reflag_fn(df)
    
        # ---- selection prompt ----
        prompt = (
            f"[ID {df.at[idx,'ID']}] Edit which row? "
            "[c]urrent / [p]rev / [n]ext / [s]kip / [o]k / [q]uit "
            "or type 'id 32444' / 'idx 32793' (or just a number) to edit a different row: "
        )
        choose = input(prompt).strip().lower()
    
        if choose == 'q':
            print("Stopping review.")
            break
        if choose in ('s', ''):
            continue
        
        # NEW: Handle 'ok' - add to skip list
        if choose == 'o':
            if 'ID' in df.columns:
                _add_to_skip_list(df.at[idx, 'ID'])
            else:
                print("Cannot add to skip list: 'ID' column not found.")
            continue
    
        # parse free-form choices
        target = None
        if choose in ('c', 'p', 'n'):
            if choose == 'p':
                if prev_idx is None:
                    print("No previous row in this logbook.")
                    continue
                target = prev_idx
            elif choose == 'n':
                if next_idx is None:
                    print("No next row in this logbook.")
                    continue
                target = next_idx
            else:
                target = idx
        else:
            # accept: 'id 32444' or 'idx 32793' or just '32444'
            tokens = choose.split()
            if len(tokens) == 2 and tokens[0] in ('id', 'idx') and tokens[1].isdigit():
                kind, num = tokens[0], int(tokens[1])
                if kind == 'id':
                    matches = df.loc[df_log.index][df.loc[df_log.index, 'ID'] == num]
                    if matches.empty:
                        print(f"ID {num} not found in this logbook subset.")
                        continue
                    target = matches.index[0]
                else:  # 'idx'
                    if num not in df_log.index:
                        print(f"Index {num} not in this logbook subset.")
                        continue
                    target = num
            elif choose.isdigit():
                num = int(choose)
                matches = df.loc[df_log.index][df.loc[df_log.index, 'ID'] == num]
                if not matches.empty:
                    target = matches.index[0]
                elif num in df_log.index:
                    target = num
                else:
                    print(f"'{choose}' not recognized as ID or index in this logbook.")
                    continue
            else:
                print("Input not recognized. Skipping.")
                continue
    
            # optional: confirm visually by starring the chosen target
            _ = plot_logbook(
                df=df,
                logbook_id=logbook_id,
                years=years,
                year_range=year_range,
                only_new=only_new,
                new_rows=new_rows,
                annotate=annotate,
                annotate_field=annotate_field,
                annotate_max=annotate_max,
                projection=projection,
                marker=marker,
                figures_dir=figures_dir,
                filename=filename,
                title_extra=title_extra,
                dpi=dpi,
                highlight_idx=target,
            )
    
        # which column?
        col_choice = input("Edit which column? [lat] / [lon] / [both] / [o]k (Enter=skip): ").strip().lower()
        
        # NEW: Allow 'ok' at column choice too
        if col_choice == 'o':
            if 'ID' in df.columns:
                _add_to_skip_list(df.at[target, 'ID'])
            else:
                print("Cannot add to skip list: 'ID' column not found.")
            continue
        
        if not col_choice:
            print("Skipped.")
            continue
    
        _edit_and_refresh(target, col_choice)

    print("\nDone reviewing flagged entries.")
    return df
