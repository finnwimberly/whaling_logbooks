import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from datetime import date, datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd
from typing import Optional, Tuple

current_directory = os.getcwd()
Figures = os.path.join(current_directory, 'Figures')

#SINGLE LOGBOOK TRAJECTORY PLOT
def plot_single_journey(
    df,
    logbook_id,
    *,
    save = False,
    global_view=True,
    bounds=None,                 # (lon_min, lon_max, lat_min, lat_max) in degrees
    projection=ccrs.Robinson(),  # any Cartopy projection
    southernmost=False,        # optional: force-highlight a specific ID
    linewidth=1.2,
    markersize=3,
    title=None,
):
    """
    Plot the full voyage path for a single logbook, with start/end and southernmost point highlighted.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'LogBook ID', 'Latitude_decimal', 'Longitude_decimal', 'DateTime', 'ID'.
    logbook_id : str
        Exact value from the 'LogBook ID' column to plot.
    global_view : bool, default True
        If True, show a global Robinson map. If False, use `bounds` if provided, otherwise auto-zoom to data.
    bounds : tuple(lon_min, lon_max, lat_min, lat_max), optional
        Map extent in PlateCarree degrees. Used only when global_view=False.
    projection : cartopy.crs, default ccrs.Robinson()
        The map projection to draw.
    southernmost_id : int, optional
        If provided, highlight this ID as the southernmost point instead of auto-computing.
    linewidth : float, default 1.2
        Line width for the voyage path.
    markersize : float, default 3
        Marker size for the path points.
    title : str, optional
        Custom title. If None, uses a default based on `logbook_id`.

    Returns
    -------
    fig, ax, info : (matplotlib.figure.Figure, matplotlib.axes.Axes, dict)
        `info` includes rows for 'start', 'end', and 'south' (southernmost) as pandas Series.
    """
    # --- filter & sort ---
    cols_needed = {'LogBook ID', 'Latitude_decimal', 'Longitude_decimal', 'Entry Date Time', 'ID'}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    voyage = (
        df[(df['LogBook ID'] == logbook_id)
           & df['Latitude_decimal'].notna()
           & df['Longitude_decimal'].notna()]
        .sort_values('Entry Date Time')
        .copy()
    )
    if voyage.empty:
        raise ValueError(f"No valid rows found for: {logbook_id}")

    # --- key points ---
    start_row = voyage.iloc[0]
    end_row   = voyage.iloc[-1]
    south_row = voyage.loc[voyage['Latitude_decimal'].idxmin()]

    lats = voyage['Latitude_decimal'].to_numpy()
    lons = voyage['Longitude_decimal'].to_numpy()

    # --- figure & map ---
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': projection})

    if global_view:
        ax.set_global()
    else:
        if bounds is None:
            # Auto-zoom with padding
            pad_lat = max(2.0, 0.1 * (np.nanmax(lats) - np.nanmin(lats) or 1))
            pad_lon = max(2.0, 0.1 * (np.nanmax(lons) - np.nanmin(lons) or 1))
            lat_min, lat_max = np.nanmin(lats) - pad_lat, np.nanmax(lats) + pad_lat
            lon_min, lon_max = np.nanmin(lons) - pad_lon, np.nanmax(lons) + pad_lon
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        else:
            lon_min, lon_max, lat_min, lat_max = bounds
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.LAND, facecolor='#b0b0b0')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.4, linestyle='--')

    # --- draw path ---
    ax.plot(
        lons, lats,
        transform=ccrs.Geodetic(),   # handles dateline nicely
        color='steelblue',
        linewidth=linewidth,
        marker='o',
        markersize=markersize,
        label='Voyage path'
    )

    # --- highlight points ---
    # ax.plot(
    #     start_row['Longitude_decimal'], start_row['Latitude_decimal'],
    #     transform=ccrs.PlateCarree(),
    #     marker='^', markersize=10, color='green', label=f"Start (ID {int(start_row['ID'])})"
    # )
    # ax.plot(
    #     end_row['Longitude_decimal'], end_row['Latitude_decimal'],
    #     transform=ccrs.PlateCarree(),
    #     marker='s', markersize=10, color='black', label=f"End (ID {int(end_row['ID'])})"
    #)
    if southernmost == True:
        ax.plot(
            south_row['Longitude_decimal'], south_row['Latitude_decimal'],
            transform=ccrs.PlateCarree(),
            marker='*', markersize=12, color='red',
            label=f"Southernmost point (Lat = {south_row['Latitude']})"
        )

    # --- title/legend ---
    if title is None:
        title = f"Voyage of {logbook_id}"
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=8)

    plt.tight_layout()
    # Save
    if save:
        os.makedirs(Figures, exist_ok=True)
        stamp = datetime.today().date().isoformat()
        if global_view:
            region = 'global'
        else:
            region = bounds
        out = os.path.join(Figures, f'Tier4_Usable_{logbook_id}_region_{stamp}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax



#HELPERS FOR GLOBAL/DECADAL CONSTRAINED PLOTS
_PRESET_BOUNDS = {
    # (lon_min, lon_max, lat_min, lat_max) in degrees
    "global": None,
    "north_atlantic": (-85, 20, 0, 70),
    "south_atlantic": (-70, 20, -60, 0),
    "indian_ocean": (20, 130, -70, 30), 
    "pacific": (110, -80, -60, 60), 
    "pacific_ne": (-160, -110, 30, 65),
    "cape_cod": (-73.5, -65.0, 38.0, 45.0),
    "pnw": (-150, -115, 35, 60),
    "southern_ocean": (-180, 180, -75, -40),
}

def _resolve_bounds(region: str | None, bounds: tuple | None):
    """Return (bounds, proj) where bounds may be None for global.
    If the box crosses the antimeridian, use PlateCarree for a stable extent."""
    if bounds is None and region:
        bounds = _PRESET_BOUNDS.get(region.lower(), None)

    # Choose projection: Robinson for global or wide views, PlateCarree for regional boxes
    if bounds is None:
        proj = ccrs.Robinson(-60)  # your original center
    else:
        lon_min, lon_max, _, _ = bounds
        crosses_dateline = (lon_min > lon_max)  # simple heuristic
        proj = ccrs.PlateCarree() if crosses_dateline else ccrs.PlateCarree()
    return bounds, proj

def _apply_extent(ax, bounds):
    if bounds is None:
        ax.set_global()
    else:
        lon_min, lon_max, lat_min, lat_max = bounds
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

def _add_base_map(ax, draw_labels=False):
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='#b0b0b0')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=draw_labels,
                      linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
    if draw_labels:
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 12}
        gl.ylabel_style = {'size': 12}
    return gl

# -------------------------
# 1) Global (now: Global or ROI)
# -------------------------
def plot_global(df, Figures,
    start_year=1820, end_year=1890, save=True,
    *, region: str | None = None, bounds: tuple | None = None,
    title: str | None = None
):
    """
    Plot usable entries color-coded by year, either globally (default) or within a region.

    Args:
        df_coords (pd.DataFrame): Has 'Entry Date Time'.
        df_usable (pd.DataFrame): Has 'Entry Date Time', 'Latitude_decimal', 'Longitude_decimal'.
        Figures (str): Output folder.
        start_year, end_year (int): Year filter inclusive.
        save (bool): Save PNG.
        region (str|None): One of _PRESET_BOUNDS keys (e.g., 'pnw', 'cape_cod', 'pacific', 'global'), or None.
        bounds (tuple|None): (lon_min, lon_max, lat_min, lat_max). Overrides `region` if given.
        title (str|None): Custom figure title.
    """
  
    df_usable = df[df['Entry Date Time'].dt.year.between(start_year, end_year, inclusive='both')]

    years = sorted(df_usable['Entry Date Time'].dt.year.dropna().unique())
    if not years:
        raise ValueError("No usable entries within the requested date range.")

    norm = mcolors.Normalize(vmin=int(min(years)), vmax=int(max(years)))
    cmap = plt.colormaps['viridis']

    # Region / projection
    bounds, proj = _resolve_bounds(region, bounds)

    # Figure
    fig, ax = plt.subplots(figsize=(19, 12), subplot_kw={'projection': proj})
    _apply_extent(ax, bounds)
    _add_base_map(ax, draw_labels=True if bounds is not None else True)

    # Plot: color-coded by year
    for yr in years:
        work = df_usable[df_usable['Entry Date Time'].dt.year == yr]
        if not len(work):
            continue
        ax.plot(
            work['Longitude_decimal'], work['Latitude_decimal'], '+',
            transform=ccrs.PlateCarree(),
            color=cmap(norm(int(yr))), markersize=4, mew=1
        )

    # Colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.03, fraction=0.05)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Year', fontsize=16, labelpad=10)

    # Title
    if title is None:
        region_name = region if region else ('global' if bounds is None else 'custom region')
        title = f'All Usable Entries'
    plt.title(title, fontsize=18, y=1.03)

    # Save
    if save:
        os.makedirs(Figures, exist_ok=True)
        stamp = datetime.today().date().isoformat()
        tag = (region or 'global') if bounds is None else 'custom_bounds'
        out = os.path.join(Figures, f'Tier4_Usable_{start_year}_{end_year}_{tag}_{stamp}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax

def _coerce_to_timestamp(val, role: str) -> pd.Timestamp:
    """
    Convert user-supplied date-like input to a pandas Timestamp.
    role: 'start' or 'end' (controls how year/month-only inputs are closed).
    Accepted: int (YYYY), str ('YYYY'|'YYYY-MM'|'YYYY-MM-DD'), datetime/date, Timestamp.
    """
    if val is None:
        return None

    # Already a Timestamp
    if isinstance(val, pd.Timestamp):
        return val.tz_localize(None) if val.tzinfo else val

    # datetime / date
    if isinstance(val, (datetime, date)):
        return pd.Timestamp(val)

    # int -> interpret as year
    if isinstance(val, int):
        if role == 'start':
            return pd.Timestamp(f"{val}-01-01 00:00:00")
        else:
            # end-of-year inclusive
            return pd.Timestamp(f"{val}-12-31 23:59:59")

    # strings
    s = str(val).strip()
    # handle bare 'YYYY'
    if len(s) == 4 and s.isdigit():
        return _coerce_to_timestamp(int(s), role)

    # handle 'YYYY-MM' month-only
    if len(s) == 7 and s[4] == '-':
        if role == 'start':
            return pd.Timestamp(f"{s}-01 00:00:00")
        else:
            # end-of-month inclusive: take first of next month minus 1 second
            ts = pd.Timestamp(f"{s}-01")
            next_month = (ts + pd.offsets.MonthBegin(1))
            return next_month - pd.Timedelta(seconds=1)

    # else fall back to pandas parsing (YYYY-MM-DD, etc.)
    ts = pd.to_datetime(s, errors='raise')
    # if it parsed to a DatetimeIndex (unlikely here), take first element
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    return ts.tz_localize(None) if getattr(ts, "tzinfo", None) else ts


def plot_time_range(
    df: pd.DataFrame,
    Figures: str,
    start=None,
    end=None,
    *,
    color: str = "tab:blue",
    marker: str = "+",
    markersize: int = 4,
    mew: float = 1.0,
    bounds: Optional[Tuple[float, float, float, float]] = None,  # (lon_min, lon_max, lat_min, lat_max)
    projection: ccrs.Projection = ccrs.Robinson(),
    title: Optional[str] = None,
    save: bool = True,
):
    """
    Plot all usable entries in a single color for a flexible time range.

    Requires df columns:
      - 'Entry Date Time' (string or datetime-like)
      - 'Latitude_decimal' (float)
      - 'Longitude_decimal' (float)

    start/end can be:
      - int year (e.g., 1820)
      - 'YYYY', 'YYYY-MM', 'YYYY-MM-DD'
      - datetime/date/pandas Timestamp
      - None (defaults to full span in data)
    """
    if 'Entry Date Time' not in df.columns:
        raise ValueError("df must contain 'Entry Date Time'.")

    # Ensure datetime dtype
    if not pd.api.types.is_datetime64_any_dtype(df['Entry Date Time']):
        df = df.copy()
        df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')

    # Coerce inputs (None -> use data min/max)
    start_ts = _coerce_to_timestamp(start, role='start') if start is not None else df['Entry Date Time'].min()
    end_ts   = _coerce_to_timestamp(end, role='end')     if end   is not None else df['Entry Date Time'].max()

    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Unable to determine valid start/end timestamps (check inputs and 'Entry Date Time' values).")
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts  # swap

    # Filter usable rows
    mask = (
        df['Entry Date Time'].between(start_ts, end_ts, inclusive='both')
        & df['Latitude_decimal'].notna()
        & df['Longitude_decimal'].notna()
    )
    dff = df.loc[mask]
    if dff.empty:
        raise ValueError("No entries found within the requested time range and with valid coordinates.")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 9), subplot_kw={'projection': projection})

    # Extent (optional)
    if bounds is not None:
        lon_min, lon_max, lat_min, lat_max = bounds
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    # Lightweight base map
    ax.coastlines(linewidth=0.6)
    try:
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    except Exception:
        ax.gridlines(linewidth=0.3, alpha=0.5)

    # Points 
    ax.plot(
        dff['Longitude_decimal'].values,
        dff['Latitude_decimal'].values,
        marker,
        transform=ccrs.PlateCarree(),
        color=color,
        markersize=markersize,
        mew=mew,
        linestyle="None",
    )

    # Title
    if title is None:
        title = f"All entries from {start_ts.date()} to {end_ts.date()} (n={len(dff)})"
    ax.set_title(title, fontsize=16, pad=10)

    # Save
    if save:
        os.makedirs(Figures, exist_ok=True)
        start_str = pd.Timestamp(start_ts).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_ts).strftime("%Y%m%d")
        out = os.path.join(Figures, f"Entries_{start_str}_{end_str}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight')

    plt.show()
    return fig, ax

# -------------------------
# 2) Decadal (now: Global or ROI)
# -------------------------
def plot_decadal_compact(
    df, Figures, start_decade=1820, end_decade=1900, save=False,
    *, region: str | None = None, bounds: tuple | None = None,
    title: str | None = None
):
    """
    Plot Tier 4 usable entries by decade in subplots with minimal horizontal spacing.
    Notes:
      - Uses a custom GridSpec (wspace=0.0) for near-zero horizontal gaps.
      - Only left-column axes show y labels; only bottom-row axes show x labels.
    """
    # --- prep time fields ---
    df = df.copy()
    df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')
    df['Year']   = df['Entry Date Time'].dt.year
    df['Decade'] = (df['Year'] // 10) * 10
    # --- filter decades ---
    decades = list(range(start_decade, end_decade, 10))
    df_f = df.loc[df['Decade'].isin(decades)].copy()
    # --- color scaling ---
    if len(df_f):
        vmin, vmax = int(df_f['Year'].min()), int(df_f['Year'].max())
    else:
        vmin, vmax = start_decade, end_decade - 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['viridis']
    # --- region / projection ---
    # Assumes you have resolve_bounds & apply_extent defined elsewhere.
    bounds, proj = _resolve_bounds(region, bounds)
    # --- layout with constrained_layout ---
    num_decades = len(decades)
    ncols = 2
    nrows = (num_decades + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(14, 6 * max(nrows, 1)),
        layout='constrained',
        subplot_kw={'projection': proj}
    )
    # Convert axes to 1D list for easier iteration
    axes = axes.flatten() if num_decades > 1 else [axes]
    # --- title ---
    if title is None:
        region_name = region if region else ('global' if bounds is None else 'custom region')
        title = f'Usable Entries by Decade'
    fig.suptitle(title, fontsize=24, y=1.03)
    # --- plot per panel ---
    for i, decade_start in enumerate(decades):
        if i >= len(axes): break
        ax = axes[i]
        # # Only left column gets y-labels; only bottom row gets x-labels
        # r, c = divmod(i, ncols)
        # draw_labels = ((c == 0) or (r == nrows - 1))  # show some labels to aid reading
        _apply_extent(ax, bounds)
        _add_base_map(ax, draw_labels=True)
        decade_data = df_f[df_f['Decade'] == decade_start]
        if not decade_data.empty:
            ax.scatter(
                decade_data['Longitude_decimal'],
                decade_data['Latitude_decimal'],
                c=decade_data['Year'],
                cmap=cmap, norm=norm,
                s=8, marker='+',
                transform=ccrs.PlateCarree()
            )
        ax.set_title(f"{decade_start}s", fontsize=22, pad=6)
    # Hide any unused axes (if odd number of decades)
    for j in range(len(decades), len(axes)):
        fig.delaxes(axes[j])
    # --- shared colorbar at bottom ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', pad=0.02, aspect=40)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Year', fontsize=14)
    # --- save/show ---
    if save:
        os.makedirs(Figures, exist_ok=True)
        stamp = datetime.today().date().isoformat()
        tag = (region or 'global') if bounds is None else 'custom_bounds'
        out = os.path.join(Figures, f'Tier4_usable_entries_decadal_{start_decade}s-{end_decade-10}s_{tag}_{stamp}.png')
        plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, axes
