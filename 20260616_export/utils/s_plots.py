import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
from datetime import datetime
import pandas as pd


def plot_global(df_coords, df_usable, out_dir, fname, title, start_year=1820, end_year=1890, save = True):
    """
    Generates and saves a global plot of usable entries, color-coded by year.

    Args:
        df_coords (pd.DataFrame): DataFrame containing coordinates and 'Entry Date Time'.
        df_usable (pd.DataFrame): DataFrame with usable entries and 'Entry Date Time'.
        out_dir (str): Path to the directory where the figure will be saved.
        fname (str): file name. 
        start_year (int): The start year for the date filter.
        end_year (int): The end year for the date filter.
    """
    # filter for specified date range
    df_coords_date_restricted = df_coords[(df_coords['Entry Date Time'].dt.year >= start_year) & (df_coords['Entry Date Time'].dt.year <= end_year)]
    df_usable_date_restricted = df_usable[(df_usable['Entry Date Time'].dt.year >= start_year) & (df_usable['Entry Date Time'].dt.year <= end_year)]

    # Set up colormap and normalization
    years = sorted(df_usable_date_restricted['Entry Date Time'].dt.year.dropna().unique())
    norm = mcolors.Normalize(vmin=min(years), vmax=max(years))
    cmap = plt.colormaps['viridis']

    # Set up figure
    fig, ax = plt.subplots(figsize=(19, 20), subplot_kw={'projection': ccrs.Robinson(-60)})
    ax.set_global()
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#b0b0b0')

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    gl.top_labels = False
    gl.right_labels = False

    # Plot Tier 4 data color-coded by year
    for yr in years:
        work = df_usable_date_restricted[df_usable_date_restricted['Entry Date Time'].dt.year == yr]
        color = cmap(norm(yr))
        ax.plot(work['Longitude_decimal'], work['Latitude_decimal'], '+',
                transform=ccrs.PlateCarree(), color=color)

    #subplot label
    ax.text(-0.05, 1.05, 'a', transform=ax.transAxes, 
        fontsize=24, fontweight='bold', va='top', ha='right')
    
    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.03, fraction=0.05)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Year', fontsize=18, labelpad=10)

    # Titles and label
    # plt.suptitle(f'Tier 4 Usable Entries: {len(df_usable_date_restricted)}', fontsize=20, y=0.595)
    # plt.title(f'{len(df_coords_date_restricted)} entries with lat & lon using 1–5 day interpolation', fontsize=16, y=1.048)

    plt.title(f'{title}', fontsize=16, y=1.048)

    # Save figure
    stamp = datetime.today().date().isoformat()
    if save:
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, fname)
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()


def plot_decadal(df_u4, Figures, start_decade=1820, end_decade=1900, save=True):

    # Work on a real copy
    df = df_u4.copy()

    # Ensure datetime and add Year/Decade
    df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')
    df['Year']   = df['Entry Date Time'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10

    # Filter once, then reuse df_f everywhere
    decades = list(range(start_decade, end_decade, 10))  # e.g., 1820..1890
    df_f = df.loc[df['Tier4_usable'] & df['Decade'].isin(decades)].copy()

    # Colormap across actual years in the filtered data (or fix range if you prefer)
    if len(df_f):
        vmin, vmax = int(df_f['Year'].min()), int(df_f['Year'].max())
    else:
        vmin, vmax = start_decade, end_decade - 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['viridis']

    # Subplots
    num_decades = len(decades)
    ncols = 2
    nrows = (num_decades + ncols - 1) // ncols

    fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(22, 6 * max(nrows, 1)),
    subplot_kw={'projection': ccrs.Robinson()},
    constrained_layout=True,             
    gridspec_kw={'wspace': 0.06, 'hspace': 0.12}  
    )
    axes = np.atleast_1d(axes).ravel()
    
    # Add a suptitle with a little extra headroom
    fig.suptitle(
        f'Tier 4 Usable Entries by Decade ({start_decade}s–{end_decade-10}s)',
        fontsize=24
    )
    
    # Optional: tweak global padding/margins
    fig.set_constrained_layout_pads(
        w_pad=0.02,  # space between figure edge and subplots (width)
        h_pad=0.02,  # space between figure edge and subplots (height)
        wspace=0.06, # space between subplots (width)
        hspace=0.08  # space between subplots (height)
    )

    total_points = 0

    for i, decade_start in enumerate(decades):
        ax = axes[i]
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='#b0b0b0')
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.top_labels = False
        gl.right_labels = False

        letter = chr(ord('b') + i)  # b, c, d, ...
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')


        # Use df_f for the current decade
        decade_data = df_f[df_f['Decade'] == decade_start]
        print(len(decade_data))
        total_points += len(decade_data)

        if not decade_data.empty:
            # Vectorized scatter is faster than looping with ax.plot
            ax.scatter(
                decade_data['Longitude_decimal'],
                decade_data['Latitude_decimal'],
                c=decade_data['Year'],
                cmap=cmap, norm=norm,
                s=8, marker='+', transform=ccrs.PlateCarree()
            )

        ax.set_title(f"{decade_start}s", fontsize=30)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Super title
    plt.suptitle(
        f'Tier 4 Usable Entries by Decade ({start_decade}s–{end_decade-10}s)',
        fontsize=28, y=1.03
    )

    #plt.tight_layout()
    if save:
        os.makedirs(Figures, exist_ok=True)
        out = os.path.join(Figures, f'Tier4_usable_entries_decadal_{start_decade}s-{end_decade-10}s.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()

def discrete_from_cmap(cmap_name, labels, span=(0.12, 0.88)):
    """Return {label: RGBA} by sampling a continuous cmap evenly within a safe span."""
    lo, hi = span
    cmap = cm.get_cmap(cmap_name)
    xs = np.linspace(lo, hi, len(labels))
    return dict(zip(labels, [cmap(x) for x in xs]))


def plot_seasonal(input_df, Figures, save=True):
    """
    One subplot per season (DJF, MAM, JJA, SON), formatted to match plot_decadal.
    """

    # Work on a copy (avoid SettingWithCopyWarning)
    df = input_df.copy()
    df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')

    # Month / Season
    df['Month'] = df['Entry Date Time'].dt.month
    def get_season(m):
        if m in (12, 1, 2):  return 'DJF'
        if m in (3, 4, 5):   return 'MAM'
        if m in (6, 7, 8):   return 'JJA'
        if m in (9, 10, 11): return 'SON'
        return np.nan
    df['Season'] = df['Month'].apply(get_season)

    # Filter once, reuse
    df_f = df.loc[df['Tier4_usable'] & df['Season'].notna()].copy()


    season_order  = ['DJF', 'MAM', 'JJA', 'SON']
    season_colors = {
        'DJF': to_hex(get_cmap('viridis')(0.20)),  # cool blue (winter)
        'MAM': to_hex(get_cmap('viridis')(0.74)),  # a bit more yellow-green
        'JJA':  to_hex(get_cmap('inferno')(0.55)),
        'SON': to_hex(get_cmap('plasma')(0.76)),  
    }


    # --- Subplots & spacing to MATCH plot_decadal ---
    ncols, nrows = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(22, 6 * max(nrows, 1)),
        subplot_kw={'projection': ccrs.Robinson()},   # match plot_decadal projection
        constrained_layout=True,
        gridspec_kw={'wspace': 0.06, 'hspace': 0.12}  # match plot_decadal
    )
    axes = np.atleast_1d(axes).ravel()

    total_points = 0
    for i, season in enumerate(season_order):
        ax = axes[i]
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor='#b0b0b0')

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=2, color='gray', alpha=0.5, linestyle='--')

        letter = chr(ord('b') + i)  # b, c, d, ...
        ax.text(-0.05, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right')
        
        gl.xlabel_style = {'size': 14}
        gl.ylabel_style = {'size': 14}
        gl.top_labels = False
        gl.right_labels = False

        sdata = df_f.loc[df_f['Season'] == season]
        total_points += len(sdata)
        print(len(sdata))

        if not sdata.empty:
            ax.scatter(
                sdata['Longitude_decimal'],
                sdata['Latitude_decimal'],
                s=8, marker='+', alpha=0.85,
                color=season_colors.get(season, 'k'),
                transform=ccrs.PlateCarree()
            )

        ax.set_title(f"{season}", fontsize=30)  # match plot_decadal axis title size

    # Delete any unused axes (shouldn't be needed with 4 seasons, but safe)
    for j in range(len(season_order), len(axes)):
        fig.delaxes(axes[j])

    # Same suptitle sizing/offset as plot_decadal
    plt.suptitle(
        f"Tier 4 Usable Entries by Season",
        fontsize=28, y=1.05
    )

    # Same global padding tweaks as plot_decadal
    fig.set_constrained_layout_pads(
        w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.08
    )

    if save:
        os.makedirs(Figures, exist_ok=True)
        out = os.path.join(Figures, 'Tier4_by_season.png')
        fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()


def plot_infill_boxplot_map(df, figures_path, save=True):
    """
    Generates a composite figure showing gap-distance distributions (ALL gaps) and a map of infilled points.
    Expects a single dataframe with:
      - gap_days_missing, gap_distance_km
      - Infilled (bool), infill_days_missing
      - Longitude_decimal, Latitude_decimal
    """
    # --- organize data ---
    day_map = {
        1: 'One Day Dist.', 2: 'Two Day Dist.', 3: 'Three Day Dist.',
        4: 'Four Day Dist.', 5: 'Five Day Dist.'
    }

    # Boxplot data: ALL gaps with distances + tier
    df_gaps = df.copy()
    df_gaps = df_gaps[df_gaps['gap_distance_km'].notna() & df_gaps['gap_days_missing'].isin(day_map.keys())]
    df_gaps['Type'] = df_gaps['gap_days_missing'].map(day_map)

    # Map data: ONLY successfully infilled points
    df_map = df[df['Infilled']].copy()
    df_map = df_map[df_map['infill_days_missing'].isin(day_map.keys())]
    df_map['Type'] = df_map['infill_days_missing'].map(day_map)

    day_labels = ['One Day Dist.', 'Two Day Dist.', 'Three Day Dist.', 'Four Day Dist.', 'Five Day Dist.']
    day_palette = discrete_from_cmap('plasma', day_labels)
    
    # --- build fig (MATCH seasonal spacing) ---
    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1, 2],
        wspace=0.06, hspace=0.02 
    )

    # subplot A: boxplot (ALL gaps by tier)
    ax1 = fig.add_subplot(gs[0])
    sns.boxplot(
        x='Type', y='gap_distance_km',
        data=df_gaps, ax=ax1,
        order=day_labels, palette=day_palette, hue='Type',
        showfliers=False
    )

    ax1.set_title('Distribution of Gaps Between Noted Coordinates', fontsize=18, y=1.05)
    ax1.set_ylabel('Distance (km)', fontsize=16)
    ax1.set_xlabel('')
    ax1.set_ylim(-100, 1200)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Threshold lines (keys must match day_labels exactly) — keep your policy values
    thresholds = {
        'One Day Dist.': 350, 'Two Day Dist.': 400, 'Three Day Dist.': 450,
        'Four Day Dist.': 450, 'Five Day Dist.': 450
    }
    for i, label in enumerate(day_labels):
        ax1.hlines(thresholds[label], xmin=i-0.4, xmax=i+0.4,
                   color='black', linestyle='--', zorder=5)

    # n counts (printed)
    for label in day_labels:
        count = df_gaps[df_gaps['Type'] == label].shape[0]
        print(f'n={count}')

    ax1.text(-0.05, 1.07, 'a', transform=ax1.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # subplot B: Map (INFILLED ONLY)
    ax2 = fig.add_subplot(gs[1], projection=ccrs.Robinson(-60))
    ax2.set_global()
    ax2.coastlines(linewidth=0.5)
    ax2.add_feature(cfeature.LAND, facecolor='#b0b0b0')

    gl = ax2.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=2, color='gray', alpha=0.5, linestyle='--'
        )
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    gl.top_labels = False
    gl.right_labels = False

    # Plot each day's infilled points
    for label, color in day_palette.items():
        df_day = df_map[df_map['Type'] == label]
        if not df_day.empty:
            ax2.plot(
                df_day['Longitude_decimal'], df_day['Latitude_decimal'],
                'o', markersize=4,
                transform=ccrs.PlateCarree(),
                label=f'{label.replace(" Dist.", "")}',
                color=color
            )

    ax2.set_title('Infilled Entries', fontsize=18, y=1.05)
    legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03),
                        ncol=5, fontsize=16)
    for handle in legend.legend_handles:
        handle.set_markersize(10)

    fig.text(0, 0.6, 'b', fontsize=18, fontweight='bold', va='top', ha='left')

    # Match the seasonal plot's padding tweaks
    fig.set_constrained_layout_pads(
        w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.02
    )

    if save:
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, 'Infilling_Report.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()

def plot_infill_boxplot_map_split(df, figures_path, save=True):
    """
    Boxplot of ALL gaps split into Paired (lat+lon) vs Single (lat OR lon) per tier.
    SAME 5 day colors as the non-split version; Single drawn lighter (alpha=0.5).
    Bottom panel maps ONLY infilled entries, colored by tier.
    """
    import matplotlib as mpl
    import matplotlib.colors as mcolors

    # --- organize data ---
    day_map = {1:'One Day Dist.', 2:'Two Day Dist.', 3:'Three Day Dist.', 4:'Four Day Dist.', 5:'Five Day Dist.'}
    day_labels  = ['One Day Dist.', 'Two Day Dist.', 'Three Day Dist.', 'Four Day Dist.', 'Five Day Dist.']

    # --- build a custom palette that handles both day and type (Paired/Single) ---
    # The palette will have keys like ('One Day Dist.', 'Paired')
    custom_palette = {}
    day_palette = discrete_from_cmap('plasma', day_labels)
    for day_label, base_color in day_palette.items():
        # Paired is fully opaque
        custom_palette[(day_label, 'Paired')] = mcolors.to_rgba(base_color, 1.0)
        # Single has 50% alpha
        custom_palette[(day_label, 'Single')] = mcolors.to_rgba(base_color, 0.5)

    # build data
    df_gaps = df.copy()
    df_gaps = df_gaps[df_gaps['gap_distance_km'].notna() & df_gaps['gap_days_missing'].isin(day_map.keys())]
    df_gaps['Type'] = df_gaps['gap_days_missing'].map(day_map)
    df_gaps['CoordKind'] = df_gaps['gap_type'].map({'latlon':'Paired', 'lat':'Single', 'lon':'Single'})

    df_map = df[df['Infilled']].copy()
    df_map = df_map[df_map['infill_days_missing'].isin(day_map.keys())]
    df_map['Type'] = df_map['infill_days_missing'].map(day_map)

    # --- figure ---
    fig = plt.figure(figsize=(18, 18), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], wspace=0.06, hspace=0.02)

    # A) grouped boxplot
    ax1 = fig.add_subplot(gs[0])
    hue_order = ['Paired', 'Single']

    # draw with a neutral palette; we’ll recolor manually so day colors carry across hue
    sns.boxplot(
        x='Type', y='gap_distance_km', hue='CoordKind',
        data=df_gaps, order=day_labels, hue_order=hue_order,
        ax=ax1, showfliers=False, dodge=True, palette=['#999999', '#999999'], legend=False
    )

    # Recolor patches assuming seaborn groups all of one hue category, then the next
    patch_i = 0
    patches = ax1.patches
    for hue_label in hue_order:
        for day_label in day_labels:
            # Check if this group has data and thus a patch was created
            has_data = not df_gaps[(df_gaps['Type'] == day_label) & (df_gaps['CoordKind'] == hue_label)].empty
            if has_data:
                if patch_i >= len(patches):
                    break  # Safety break
                
                patch = patches[patch_i]
                base_color = day_palette[day_label]
                alpha = 1.0 if hue_label == 'Paired' else 0.5
                
                patch.set_facecolor(mcolors.to_rgba(base_color, alpha))
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
                patch_i += 1

    for line in ax1.lines:  # whiskers/medians
        line.set_color('black')
        line.set_linewidth(1.0)

    #legend
    from matplotlib.patches import Patch
    legend_color = day_palette['One Day Dist.']
    legend_elements = [
        Patch(facecolor=mcolors.to_rgba(legend_color, 1.0), edgecolor='black', label='Coordinate Pair'),
        Patch(facecolor=mcolors.to_rgba(legend_color, 0.5), edgecolor='black', label='Single Coordinate')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=14)

    ax1.set_title('Gap Distance Distributions', fontsize=18, y=1.05)
    ax1.set_xlabel('')
    ax1.set_ylabel('Distance (km)', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    thresholds = {'One Day Dist.': 350, 'Two Day Dist.': 400, 'Three Day Dist.': 450, 'Four Day Dist.': 450, 'Five Day Dist.': 450}
    for i, label in enumerate(day_labels):
        ax1.hlines(thresholds[label], xmin=i-0.4, xmax=i+0.4, color='black', linestyle='--', zorder=5)

    ax1.text(-0.05, 1.07, 'a', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # B) map (infilled only) – same palette by day
    ax2 = fig.add_subplot(gs[1], projection=ccrs.Robinson(-60))
    ax2.set_global()
    ax2.coastlines(linewidth=0.5)
    ax2.add_feature(cfeature.LAND, facecolor='#b0b0b0')

    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    gl.top_labels = False
    gl.right_labels = False

    for label, color in day_palette.items():
        df_day = df_map[df_map['Type'] == label]
        if not df_day.empty:
            ax2.plot(df_day['Longitude_decimal'], df_day['Latitude_decimal'],
                     'o', markersize=4, transform=ccrs.PlateCarree(),
                     label=f'{label.replace(" Dist.", "")}', color=color)

    ax2.set_title('Infilled Entries', fontsize=18, y=1.05)
    legend = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=16)
    for h in legend.legend_handles: h.set_markersize(10)

    fig.text(0, 0.6, 'b', fontsize=18, fontweight='bold', va='top', ha='left')
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.06, hspace=0.02)

    if save:
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path, 'Infilling_Report_SPLIT.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_annual_distribution(df, Figures, start_year=1820, end_year=1890, save=True):
    """
    Generates and saves a plot of the annual distribution of Tier 4 entries.

    Args:
        df (pd.DataFrame): DataFrame containing 'DateTime' column.
        Data (str): Path to the directory where the figure will be saved.
        start_year (int): The start year for the filter.
        end_year (int): The end year for the filter.
    """
    # Filter out rows where 'DateTime' is missing
    df_valid_dates = df[df["Entry Date Time"].notna()].copy()

    # Extract year safely
    df_valid_dates["Year"] = df_valid_dates["Entry Date Time"].dt.year

    # Filter to specified year range
    df_valid_dates = df_valid_dates[(df_valid_dates["Year"] >= start_year) & (df_valid_dates["Year"] <= end_year)]

    # Count unique years
    num_years = df_valid_dates["Year"].nunique()

    # Generate year range for ordering
    all_years = np.arange(start_year, end_year + 1)

    # Plot setup
    f, ax = plt.subplots(figsize=(12, 8))

    # Rotated format: years on x-axis
    sns.countplot(x=df_valid_dates["Year"], order=all_years, ax=ax,
                  color='black', alpha=0.6)

    # Grid and ticks
    ax.grid(alpha=0.2)
    tick_spacing = 10
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # Axis labels and title
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Number of Entries", fontsize=18)
    plt.title(f'Number of Usable Entries per Year ({start_year}–{end_year})', fontsize=18)

    # Add subplot label 'a'
    ax.text(-0.08, 1.1, 'a', transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='right')

    # Save (optional)
    if save:
        save_dir = os.path.join(Figures, f'Tier4_Entries_Per_Year_{start_year}-{end_year}.png')
        plt.savefig(save_dir, dpi=300, bbox_inches='tight')

    plt.show()

def plot_bf_distribution(df, Figures, save=True):
    """
    Generates and saves a plot of the distribution of BF values.
    Handles float input by treating them as discrete Beaufort categories.
    """
    # 1. Drop NaNs and ensure we are working with a clean Series
    bf_data = df["BF Value"].dropna()

    # 2. Convert to int for cleaner plotting (removes the .0)
    # If you have half-steps (like 4.5), skip this line and keep them as floats
    bf_data = bf_data.astype(int)

    # 3. Define the Beaufort Scale range (0 to 12)
    # This ensures even missing categories (like 11 or 12) show up on the axis
    bf_range = np.arange(0, 13)

    # 4. Plot setup
    f, ax = plt.subplots(figsize=(10, 6))

    # 5. Create the count plot
    sns.countplot(x=bf_data, order=bf_range, ax=ax,
                  color='black', alpha=0.6)

    # 6. Grid and Aesthetics
    ax.grid(axis='y', alpha=0.2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) # Tick for every BF number

    # 7. Axis labels and title
    plt.xlabel("Beaufort Value", fontsize=16)
    plt.ylabel("Number of Entries", fontsize=16)
    plt.title('Distribution of Beaufort (BF) Values', fontsize=18)

    # 8. Add subplot label 'b'
    ax.text(-0.08, 1.1, 'b', transform=ax.transAxes, 
            fontsize=16, fontweight='bold', va='top', ha='right')

    # 9. Save (optional)
    if save:
        os.makedirs(Figures, exist_ok=True)
        save_path = os.path.join(Figures, 'Tier4_BF_Distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()