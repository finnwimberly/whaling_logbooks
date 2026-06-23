import re
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

def standardize_logbook_id(raw: str) -> str:
    """
    Standardize spacing, dash style, and vessel-type capitalization
    in logbook IDs. Works even if vessel type or years are missing.
    """
    if not isinstance(raw, str):
        return raw

    s = raw.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")

    # Pattern pieces:
    # name (required)
    # (type) (optional)
    # years (optional, single year or YYYY-YYYY)
    pattern = r"""
        ^\s*
        (?P<name>[^(0-9]+?)          # name = letters and spaces
        \s*
        (?:\(\s*(?P<type>[^)]+)\s*\))?   # optional (Ship)
        \s*
        (?P<years>(\d{4})(?:-(\d{4}))?)? # optional YYYY or YYYY-YYYY
        \s*$
    """

    m = re.match(pattern, s, re.VERBOSE)
    if not m:
        return s

    name  = m.group("name").strip()
    vtype = m.group("type")
    years = m.group("years")

    # Normalize type
    if vtype:
        type_map = {
            "ship": "Ship",
            "bark": "Bark",
            "brig": "Brig",
            "schooner": "Schooner",
            "brigantine": "Brigantine",
        }
        vtype_norm = type_map.get(vtype.lower(), vtype.capitalize())
    else:
        vtype_norm = None

    # Normalize years
    if years:
        years = re.sub(r"\s*-\s*", "-", years)
        years_norm = years
    else:
        years_norm = None

    # Reconstruct
    parts = [name]
    if vtype_norm:
        parts.append(f"({vtype_norm})")
    if years_norm:
        parts.append(years_norm)

    return " ".join(parts)


# truncated mapping dictionary:
logbook_id_fix_map = {
    "A. M. Nicholson (schooner) 190…":    "A. M. Nicholson (schooner) 1909-1910",   
    "Abraham Barker (Ship) 1850-185…":     "Abraham Barker (Ship) 1850-1853",
    "Alexander Barclay (Ship) 1837-…":     "Alexander Barclay (Ship) 1837-1838",
    "Bartholomew Gosnold (Bark) 188…":     "Bartholomew Gosnold (Bark) 1881-1885",
    "C.C. Comstock (Schooner) 1865-…":     "C.C. Comstock (Schooner) 1865-1866",
    "Charles Phelps (Ship) 1844-184…":     "Charles Phelps (Ship) 1844-1847",
    "Charles Phelps (Ship) 1850-185…":     "Charles Phelps (Ship) 1850-1853",
    "Charles Phelps (ship) 1842-184…":     "Charles Phelps (ship) 1842-1844",
    "Charles W. Morgan (bark) 1911-…":     "Charles W. Morgan (bark) 1911-1912",
    "Charles W. Morgan (Ship) 1841-…":     "Charles W. Morgan (Ship) 1841-1845 Journal",
    "Charles and Henry (ship) 1833-…":     "Charles and Henry (ship) 1833-1836",
    "Clifford Wayne (ship) 1844-184…":     "Clifford Wayne (ship) 1844-1847",
    "Eunice H. Adams (Brig) 1887-18…":     "Eunice H. Adams (Brig) 1887-1890",
    "Francis Allyn (Schooner) 1891-…":     "Francis Allyn (Schooner) 1891-1893",
    #NOTE TWO OF THESE "Gage H. Phillips (Schooner) 18…":     "",
    "General Jackson (ship) 1836-18…":     "General Jackson (ship) 1836-1839",
    "George Clinton (ship) 1834-183…":     "George Clinton (ship) 1834-1837",
    "George Washington (Bark) 1837-…":     "George Washington (Bark) 1837-1840",
    "Gideon Howland (Ship) 1838-184…":     "Gideon Howland (Ship) 1838-1842",
    "Gideon Howland (ship) 1836-183…":     "Gideon Howland (ship) 1836-1838",
    "Good Return II (ship) 1833-183…":     "Good Return II (ship) 1833-1834",
    "Governor Carver (ship) 1857-18…":     "Governor Carver (ship) 1857-1859",
    "Governor Hopkins (Brig) 1839-1…":     "Governor Hopkins (Brig) 1839-1840",
    "Henry Kneeland (ship) 1848-185…":     "Henry Kneeland (ship) 1848-1851",
    "Sally Anne (ship) & Seine…":          "Sally Anne (ship) & Seine (bark) 1835-1838",
    "Thomas Winslow (Brig) 1846-184…":     "Thomas Winslow (Brig) 1846-1847",
    "Walter Irving (Schooner) 1852-…":     "Walter Irving (Schooner) 1852-1853",
    "Walter Irving (Schooner) 1853-…":     "Walter Irving (Schooner) 1853-1854",
    "Walter Irving (Schooner) 1854-…":     "Walter Irving (Schooner) 1854-1855",
    "Walter Irving (Schooner) 1855-…":     "Walter Irving (Schooner) 1855-1856",
    "Walter Irving (Schooner) 1856-…":     "Walter Irving (Schooner) 1856-1857",
}

#SINGLE LOGBOOK TRAJECTORY PLOT

from matplotlib.colors import ListedColormap

beaufort_colors = [
    "#d0f0ff",  # 0 - calm
    "#a0e1ff",  # 1
    "#70d2ff",  # 2
    "#40c3ff",  # 3
    "#00b4ff",  # 4
    "#40c040",  # 5
    "#80d020",  # 6
    "#c0e000",  # 7
    "#f0c000",  # 8
    "#ff9000",  # 9
    "#ff6000",  # 10
    "#ff3000",  # 11
    "#cc0000",  # 12 - storm / hurricane
]

cmap_bf = ListedColormap(beaufort_colors)


def plot_logbook_with_options(
    df,
    logbook_id,
    *,
    save=False,
    save_path = None,
    selected_tier = None,
    global_view=True,
    bounds=None,                 # (lon_min, lon_max, lat_min, lat_max) in degrees
    projection=ccrs.Robinson(),  # any Cartopy projection
    color_by=None,               # 'Entry Date', 'BF Value', or a matplotlib color string
    plot_path=True,
    linewidth=1.2,
    markersize=3,
    title=None,
):
    """
    Plot the full voyage path for a single logbook, with start/end and southernmost point highlighted.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'LogBook ID', 'Latitude_decimal', 'Longitude_decimal', 'Entry Date Time', 'ID'.
    logbook_id : str
        Exact value from the 'LogBook ID' column to plot.
    global_view : bool, default True
        If True, show a global Robinson map. If False, use `bounds` if provided, otherwise auto-zoom to data.
    bounds : tuple(lon_min, lon_max, lat_min, lat_max), optional
        Map extent in PlateCarree degrees. Used only when global_view=False.
    projection : cartopy.crs, default ccrs.Robinson()
        The map projection to draw.
    color_by : str or None
        Choose variable to color points by. Either 'Entry Date', 'BF Value',
        or enter a valid matplotlib color to plot all points in same color.
        If None, uses a single default color.
    linewidth : float, default 1.2
        Line width for the voyage path.
    markersize : float, default 3
        Marker size for the path points.
    title : str, optional
        Custom title. If None, uses a default based on `logbook_id`.

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, matplotlib.axes.Axes)
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

    # Make sure datetime column is datetime
    voyage['Entry Date Time'] = pd.to_datetime(voyage['Entry Date Time'])

    # Coordinate arrays
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

    # --- draw path line (neutral color) ---
    if plot_path == True:
        path_line = ax.plot(
            lons, lats,
            transform=ccrs.Geodetic(),   # handles dateline nicely
            color='0.4',
            linewidth=linewidth,
            label='Voyage path',
        )

    # --- color points according to color_by ---
    scatter = None

    if color_by == 'Entry Date':
        # Gradient over time
        dates = voyage['Entry Date Time']
    
        # Normalize by elapsed seconds from start
        t = (dates - dates.min()).dt.total_seconds().to_numpy()
    
        if t.max() == t.min():
            # All times identical -> fall back to single color
            scatter = ax.scatter(
                lons, lats,
                transform=ccrs.Geodetic(),
                s=markersize**2,
                color='steelblue',
                zorder=3,
            )
        else:
            norm = plt.Normalize(vmin=t.min(), vmax=t.max())
            cmap = cm.get_cmap('viridis')
    
            scatter = ax.scatter(
                lons, lats,
                c=t,
                cmap=cmap,
                norm=norm,
                transform=ccrs.Geodetic(),
                s=markersize**2,
                zorder=3,
            )

            cax = fig.add_axes([0.1, 0.02, 0.8, 0.08])          
            cb = fig.colorbar(
                scatter,
                cax=cax,
                location='bottom',
                orientation='horizontal',
            )
    
            # handle date tick labels
            num_ticks = 6
            tick_vals = np.linspace(t.min(), t.max(), num_ticks)
            tick_dates = dates.min() + pd.to_timedelta(tick_vals, unit="s")
    
            cb.set_ticks(tick_vals)
            cb.set_ticklabels([d.strftime("%Y-%m-%d") for d in tick_dates])
    
            # Label showing start → end
            start_date = dates.min().strftime("%Y-%m-%d")
            end_date = dates.max().strftime("%Y-%m-%d")
            cb.set_label(f'Entry Date', fontsize = 14)

    # elif color_by == 'BF Value':
    #     if 'BF Value' not in voyage.columns:
    #         raise ValueError("Column 'BF Value' not found in DataFrame.")
    #     bf = voyage['BF Value'].to_numpy()
    #     # Discrete colormap for 0–12
    #     cmap = cm.get_cmap('plasma', 13)
    #     scatter = ax.scatter(
    #         lons, lats,
    #         c=bf,
    #         cmap=cmap_bf,
    #         vmin=0,
    #         vmax=12,
    #         transform=ccrs.Geodetic(),
    #         s=markersize**2,
    #         zorder=3,
    #     )

    #     # Create a custom colorbar axis on the right
    #     cax = fig.add_axes([1.02, 0.15, 0.04, 0.7])          
    #     cb = fig.colorbar(
    #         scatter,
    #         cax=cax,
    #         orientation='vertical'
    #     )
    #     cb.set_label('Beaufort Value')
    #     cb.set_ticks(range(0, 13))

    elif color_by == 'BF Value':
        if 'BF Value' not in voyage.columns:
            raise ValueError("Column 'BF Value' not found in DataFrame.")

        # create masks for missing vs present BF values
        bf_vals = voyage['BF Value'].to_numpy()
        has_bf = ~np.isnan(bf_vals)
        no_bf = np.isnan(bf_vals)

        # plot entries with BF values 
        scatter = ax.scatter(
            lons[has_bf], lats[has_bf],
            c=bf_vals[has_bf],
            cmap=cmap_bf,
            vmin=0,
            vmax=12,
            transform=ccrs.Geodetic(),
            s=markersize**2,
            zorder=3,
        )

        # plot entries wihtout BF values as black markers
        ax.scatter(
            lons[no_bf], lats[no_bf],
            color='black',
            label='No BF Value',
            transform=ccrs.Geodetic(),
            s=markersize**2,
            zorder=2, # slightly lower zorder 
            alpha = 0.5,
        )

        # Create a custom colorbar axis on the right
        cax = fig.add_axes([1.02, 0.15, 0.04, 0.7])          
        cb = fig.colorbar(
            scatter,
            cax=cax,
            orientation='vertical'
        )
        cb.set_label('Beaufort Value')
        cb.set_ticks(range(0, 13))

    else:
        # Treat color_by as a literal color, or default
        marker_color = color_by if isinstance(color_by, str) and color_by is not None else 'steelblue'
        scatter = ax.scatter(
            lons, lats,
            transform=ccrs.Geodetic(),
            s=markersize**2,
            color=marker_color,
            zorder=3,
            label='Positions',
        )

    # --- title/legend ---
    if title is None:
        title = f"Voyage of {logbook_id}"
    ax.set_title(title, fontsize=14, pad = 10)

    # Only add legend entry for line and (if constant color) points
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='lower left', fontsize=8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()

    # Save
    if save:
        # default directory if none provided
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "meta_figs")
        os.makedirs(save_path, exist_ok=True)
    
        stamp = datetime.today().date().isoformat()
        region = 'global' if global_view else bounds
    
        # make region readable
        if isinstance(region, tuple):
            region_str = "_".join(str(x) for x in region)
        else:
            region_str = str(region)
    
        if color_by is not None:
            safe_color = str(color_by).replace(" ", "")
            out = os.path.join(
                save_path,
                f"Tier{selected_tier}_{logbook_id}_{region_str}_{safe_color}.png"
            )
        else:
            out = os.path.join(
                save_path,
                f"Tier{selected_tier}_{logbook_id}_{region_str}.png"
            )
    
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    else:
        plt.show()
    return fig, ax


#PLOTTING GLOBAL WIND BEARING OR FORCE --------------------------------------------


def add_wind_direction_compass(fig, cmap, *,
                               vmin=0, vmax=360,
                               rect=[0.78, 0.05, 0.18, 0.18],
                               # label="Wind direction"):
                               label=""):
    """
    Add a circular compass-style legend for wind direction to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw into.
    cmap : Colormap
        Colormap used for WD_Bearing (e.g. plt.cm.twilight).
    vmin, vmax : float
        Data range for direction (usually 0–360).
    rect : [left, bottom, width, height]
        Position of the inset axes in figure coordinates.
    label : str
        Optional text label under the compass.
    """
    # Polar inset axes
    axc = fig.add_axes(rect, projection='polar')
    axc.set_theta_zero_location('N')   # 0° at North
    axc.set_theta_direction(-1)        # clockwise like a compass

    # Create a ring from r_inner to r_outer
    r_inner, r_outer = 0.6, 1.0
    theta = np.linspace(0, 2*np.pi, 361)
    r = np.linspace(r_inner, r_outer, 2)
    T, R = np.meshgrid(theta, r)

    # Map angle to data values (0–360)
    vals = np.linspace(vmin, vmax, theta.size - 1)
    V = np.tile(vals, (r.size - 1, 1))

    axc.pcolormesh(T, R, V, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")

    # Gridlines (circular & radial)
    axc.grid(True, color="black", linewidth=0.8, linestyle = '--')
    axc.spines['polar'].set_color('black')
    axc.spines['polar'].set_linewidth(0.8)

    # Tidy up: no radial ticks, only cardinal labels
    axc.set_yticklabels([])
    axc.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axc.set_xticklabels(['N', 'E', 'S', 'W'])

    # Optional label under the compass
    if label:
        # place text in figure coords roughly under the compass
        fig.text(rect[0] + rect[2]/2, rect[1] - 0.02,
                 label, ha='center', va='top')

    return axc


#plot overview figs
def plot_all_logbooks_with_options(
    df,
    *,
    save=False,
    save_path = None,
    selected_tier = None,
    global_view=True,
    bounds=None,                 # (lon_min, lon_max, lat_min, lat_max) in degrees
    projection=ccrs.Robinson(),  # any Cartopy projection
    color_by=None,               # 'Entry Date', 'BF Value', WD_Bearing, or a matplotlib color string
    markersize=3,
    title="All logbooks",
):
    """
    Plot all logbook positions in a single map, with optional coloring.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: 'LogBook ID', 'Latitude_decimal', 'Longitude_decimal',
        'Entry Date Time', 'ID'. May contain 'BF Value' for Beaufort coloring.
    save : bool, default False
        If True, saves the figure to disk.
    global_view : bool, default True
        If True, show a global Robinson map. If False, use `bounds` if provided,
        otherwise auto-zoom to data.
    bounds : tuple(lon_min, lon_max, lat_min, lat_max), optional
        Map extent in PlateCarree degrees. Used only when global_view=False.
    projection : cartopy.crs, default ccrs.Robinson()
        The map projection to draw.
    color_by : str or None
        Choose variable to color points by. Either 'BF Value' MAYBE ADD MORE LATER
        or enter a valid matplotlib color to plot all points in same color.
        If None, uses a single default color.
    markersize : float, default 3
        Marker size for the path points.
    title : str, optional
        Figure title. Default "All logbooks".

    Returns
    -------
    fig, ax : (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    # --- filter & check ---
    cols_needed = {'LogBook ID', 'Latitude_decimal', 'Longitude_decimal', 'Entry Date Time', 'ID'}
    missing = cols_needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")

    # Keep all logbooks, just drop NaN coords
    all_points = (
        df[df['Latitude_decimal'].notna() & df['Longitude_decimal'].notna()]
        .copy()
    )
    if all_points.empty:
        raise ValueError("No valid rows with lat/lon found in DataFrame.")

    # Make sure datetime column is datetime
    all_points['Entry Date Time'] = pd.to_datetime(all_points['Entry Date Time'])

    # Coordinate arrays
    lats = all_points['Latitude_decimal'].to_numpy()
    lons = all_points['Longitude_decimal'].to_numpy()

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

    scatter = None

    # --- color_by: BF Value (discrete) ---
    if color_by == 'BF Value':
        if 'BF Value' not in all_points.columns:
            raise ValueError("Column 'BF Value' not found in DataFrame.")
        bf = all_points['BF Value'].to_numpy()

        cmap = cmap_bf if 'cmap_bf' in globals() else cm.get_cmap('plasma', 13)

        scatter = ax.scatter(
            lons, lats,
            c=bf,
            cmap=cmap,
            vmin=0,
            vmax=12,
            transform=ccrs.Geodetic(),
            s=markersize**2,
            zorder=3,
        )

        # Vertical colorbar on the right, shorter + centered
        cax = fig.add_axes([1.02, 0.15, 0.04, 0.7])
        cb = fig.colorbar(
            scatter,
            cax=cax,
            orientation='vertical'
        )
        cb.set_label('Beaufort Value')
        cb.set_ticks(range(0, 13))
    
    elif color_by == "WD_Bearing":
        if "WD_Bearing" not in all_points.columns:
            raise ValueError("Column 'WD_Bearing' not found.")
    
        wd = all_points["WD_Bearing"].astype(float).to_numpy()
    
        # twilight is a cyclic colormap ideal for directions
        cmap = plt.cm.twilight
    
        scatter = ax.scatter(
            lons, lats,
            c=wd, cmap=cmap,
            vmin=0, vmax=360,
            transform=ccrs.Geodetic(),
            s=markersize**2, zorder=3,
        )
    
        # Instead of a linear colorbar, add a circular compass legend
        add_wind_direction_compass(
            fig,
            cmap=cmap,
            vmin=0,
            vmax=360,
            rect=[0.9, 0.12, 0.14, 0.14],  # tweak position/size as you like
            # label="Wind direction",
            label=""
        )

    # color_by: constant color
    else:
        marker_color = color_by if isinstance(color_by, str) and color_by is not None else 'steelblue'
        scatter = ax.scatter(
            lons, lats,
            transform=ccrs.Geodetic(),
            s=markersize**2,
            color=marker_color,
            zorder=3,
            label='Positions',
        )

    # title
    ax.set_title(title, fontsize=14, y=1.02)

    # Legend only makes sense for constant-color case
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='lower left', fontsize=8)

    # Avoid the tight_layout warning with Cartopy
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        plt.tight_layout()

    if save:
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "meta_figs")
        os.makedirs(save_path, exist_ok=True)
    
        #stamp = datetime.today().date().isoformat()
        region = 'global' if global_view else bounds
        region_str = "_".join(str(x) for x in region) if isinstance(region, tuple) else str(region)
    
        if color_by is not None:
            safe_color = str(color_by).replace(" ", "")
            out = os.path.join(save_path, f"Tier{selected_tier}_{safe_color}_{region_str}.png")
        else:
            out = os.path.join(save_path, f"Tier{selected_tier}_{safe_color}_{region_str}.png")
    
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)


    else:
        plt.show()
        
    return fig, ax
        

