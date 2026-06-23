import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm

def ms_bf_conv(dar):
    '''
    Convert m/s to Beaufort wind force scale
    
    Parameters
    ---
    dataarray of wind speed in m/s
    
    Returns
    ---
    resulting windspeed as xarray in Beaufort scale
    '''
    #converting windspeed in meters per second to Beafort scale
    darb = xr.where(dar < 0.3 , 0,
        xr.where((0.3<=dar)&(dar<=1.5),1,
                 xr.where((1.5<dar)&(dar<=3.3),2,
                         xr.where((3.3<dar)&(dar<=5.4),3,
                                 xr.where((5.4<dar)&(dar<=7.9),4,
                                         xr.where((7.9<dar)&(dar<=10.7),5,
                                                 xr.where((10.7<dar)&(dar<=13.8),6,
                                                         xr.where((13.8<dar)&(dar<=17.1),7,
                                                                 xr.where((17.1<dar)&(dar<=20.7),8,
                                                                         xr.where((20.7<dar)&(dar<=24.4),9,
                                                                                 xr.where((24.4<dar)&(dar<=28.4),10,
                                                                                         xr.where((28.4<dar)&(dar<=32.6),11,
                                                                                                 xr.where(32.6<dar,12,
                                                                                                         np.nan)))))))))))))
    return darb


def speedtoangle(uwind, vwind):
    """ 
    converts wind speeds to wind direction
    
    Parameters
    ---
    uwind: zonal wind speed
    vwind: meridional wind speed
    
    Returns
    ---
    wind direction in degrees between -180 (easterly) and 180 (westerly)
    """
    radangle = np.arctan2(uwind, vwind)
    degangle = np.degrees(radangle)
    deg360 = degangle.where(degangle.values > 0, 360 + degangle.values)
    deg360to = xr.where(deg360.values > 180, deg360 - 180, deg360 + 180)
    return deg360to


def open_model(datau, datav, latmin, latmax ,lonmin, lonmax):
    """
    open reanalysis data
    
    Parameters
    ---
    datau, datav: zonal/meridional windspeed in m/s
    latmin, latmax ,lonmin, lonmax: bounds of the desired region
    
    Returns
    ---
    xr.dataset
    """
    land = xr.open_dataset('/home/neele.sander/Downloads/land.nc')
    dau = datau.sortby('lat', ascending = True)
    dav = datav.sortby('lat', ascending = True)

        
    if np.max(dau.lon >= 181):
        dau = dau.assign_coords(lon=(((dau.lon + 180) % 360) - 180)).sortby('lon', ascending = True)

    if np.max(dav.lon >= 181):
        dav = dav.assign_coords(lon=(((dav.lon + 180) % 360) - 180)).sortby('lon', ascending = True)

    if np.max(land.lon >= 181):
        land = land.assign_coords(lon=(((land.lon + 180) % 360) - 180)).sortby('lon', ascending = True)
        
        dau = dau.sel(lat = slice(latmin, latmax), lon = slice(lonmin,lonmax))
        dav = dav.sel(lat = slice(latmin, latmax), lon = slice(lonmin,lonmax))
        landmaskmean = land.sel(lat = slice(latmin, latmax), lon = slice(lonmin,lonmax)).mean('time')
        
    dau = dau.where(landmaskmean['land'] ==0, np.nan)
    dav = dav.where(landmaskmean['land'] ==0, np.nan)
    
    #resulting wind in beaufort
    dar = np.sqrt(dau.uwnd**2 + dav.vwnd**2)
    darb = ms_bf_conv(dar).rename("speed").rename("speed")
    ## resulting angle
    deg360to = speedtoangle(dau.uwnd, dav.vwnd).rename("angle")

    ds = xr.merge([dau, dav, darb,deg360to]).mean(['nbnds']).drop_vars("time_bnds")
    
    return ds

def open_logbook(path, lonmin, lonmax, latmin, latmax):
    
    """
    open logbook data
    
    Parameters
    ---
    path: string, path to .csv file
    lonmin, lonmax, latmin, latmax: bounds of region of interest
    
    Returns
    ---
    pandas dataframe of whaling data
    """
#     with open(path, 'rb') as f:
#         data = pickle.load(f)
    data = pd.read_csv(path)
    
    # need to rename a lot because column names changed
    data = data.rename(columns={"Wind Speed/Force" :"Wind description","BF Value": "Wind Speed/Force", 
                   "Latitude" : "Latitude description", "Latitude_decimal":"Latitude",
                   "Longitude" : "Longitude description", "Longitude_decimal":"Longitude",
                  # "Entry Date Time" : "DateTime", 
                   "Wind Direction":"Wind Direction description","WD_Bearing" : "Wind Direction" })
    
    # Add year, month, and date columns using vectorized operations
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['date'] = data['DateTime'].dt.date

    # Filter data based on given latitude and longitude bounds
    datana = data[(data['Longitude'] >= lonmin) & (data['Longitude'] <= lonmax) &
                  (data['Latitude'] >= latmin) & (data['Latitude'] <= latmax)]

    # Drop NaN values and sort by date
    datana = datana.dropna(subset=['DateTime', 'Wind Speed/Force', 'Wind Direction'])
    datana = datana.sort_values(by='date', ascending=True)

    # Round longitude and latitude to create grid columns
    datana['Longitudegrid'] = datana['Longitude'].round(0)
    datana['Latitudegrid'] = datana['Latitude'].round(0)

    # Rename columns for consistency
    datana = datana.rename(columns={"Longitude": "lon", "Latitude": "lat"})
    return datana


def xrtodf(data):
    """
    convert xarray dataarray to pandas dataframe
    
    Parameters
    ---
    data: xarray dataarray
    
    Returns
    ---
    pandas dataframe
    """
    data_df = data.to_dataframe().reset_index()
    return data_df

def process_basin(model_data, binstep, startyear, endyear):
    """
    group reanalysis data by bins for specific time
    
    Parameters
    ---
    model_dara: xr.dataarray
    binstep: binstep
    startyear, endyear: floats
    
    Returns
    ---
    pd.Dataframe
    """
    bins = int((np.max(model_data.lat.values) - np.min(model_data.lat.values))/binstep)
    model_data_bin = model_data.groupby_bins('lat', bins = bins).mean().mean('lon')
    model_data_df = xrtodf(model_data_bin['speed'].groupby('time.year').mean())
    model_data_df['Latitude_bin'] = model_data_df['lat_bins'].apply(lambda x: x.mid)
    df_bin = model_data_df[(model_data_df['year'] >= startyear) & (model_data_df['year'] <= endyear) ]
    return df_bin

def month_to_season(month):
    """
    create seasons (MAM, JJA, SON, DJF) for corresponding months
    
    Parameters
    ---
    month: array of months
    
    Returns
    ---
    season
    """
    if 3 <= month <= 5:
        return 'MAM'
    elif 6 <= month <= 8:
        return 'JJA'
    elif 9 <= month <= 11:
        return 'SON'
    else:
        return 'DJF'
    
def minus_whaler(data):
    """
    Adjusts wind speeds for wind directions: 
    - If wind direction is <= 180, it assigns a negative wind speed.
    - Otherwise, keeps the wind speed positive.
    
    Parameters
    ---
    data: pd.Dataframe
    
    Returns
    ---
    pd.Dataframe
    """
    # Separate data based on wind direction
    smaller180 = data[data['Wind Direction'] <= 180].copy()
    bigger180 = data[data['Wind Direction'] > 180].copy()

    # Invert wind speed for directions <= 180
    smaller180['Wind Speed/Force'] *= -1

    # Combine the two datasets back together
    data_minus = pd.concat([smaller180, bigger180])

    return data_minus

def process_whaler(data_whaler, periods, bins, basin_name):
    """
    Process whaler data by splitting it into arbitrary periods.
    Groups data by latitudinal bins and calculates relevant statistics.
    
    Parameters
    ---
    data_whaler: DataFrame containing whaler data.
    periods: List of period boundaries (e.g., [1830, 1840, 1850, 1860]).
    bins: Latitude bins for grouping the data.
    
    Returns
    ---
    grouped data
    """
    group = {}
    

    # Add season information and reset the index
    data_whaler['season'] = data_whaler['month'].apply(month_to_season)
    data_whaler = data_whaler.reset_index(drop=True)
    if "North Atlantic" in basin_name: # no winter for North Atlantic
        data_whaler = data_whaler[(data_whaler['season'] == 'MAM') | (data_whaler['season'] == 'JJA') | (data_whaler['season'] == 'SON')]

    # Sort data by latitude
    data_whaler = data_whaler.sort_values('lat')

    # Adjust wind speed values based on wind direction
    data_whaler_minus = minus_whaler(data_whaler)

    # Ensure wind speed is treated as a float
    data_whaler_minus['speed'] = data_whaler_minus['Wind Speed/Force'].astype(float)

    # Initialize the group data with the total count and means across all periods
    group['N_total'] = len(data_whaler_minus)
    data_whaler_bins = data_whaler_minus.groupby(pd.cut(data_whaler_minus['lat'], bins)).mean().rename(columns={'lat': "lat_mean"}).reset_index()
    data_whaler_bins['lat_bin'] = data_whaler_bins['lat'].apply(lambda x: x.mid)
    group['all'] = data_whaler_bins

    # Calculate the amount per bin
    amount_bin = data_whaler_minus.groupby(pd.cut(data_whaler_minus['lat'], bins)).size().reset_index(name='amount')
    amount_bin['lat_bin'] = amount_bin['lat'].apply(lambda x: x.mid)
    group['bins'] = amount_bin[['lat_bin', 'amount']]

    # Dynamically split the data into time periods
    period_data = {}
    for i, (start, end) in enumerate(zip(periods[:-1], periods[1:])):
        period_name = f"period_{i+1}_{start}_{end}"
        period_df = data_whaler_minus[(data_whaler_minus['year'] >= start) & (data_whaler_minus['year'] < end)]
        group[f"N_{period_name}"] = len(period_df)
        period_df = period_df.groupby(pd.cut(period_df['lat'], bins)).mean().rename(columns={'lat': "lat_mean"}).reset_index()
        period_df['lat_bin'] = period_df['lat'].apply(lambda x: x.mid)
        period_data[period_name] = period_df
        group[period_name] = period_data[period_name]

    return group

def plot_whaler_data(results, basin_name, periods, ax_):
    """
    Plot the wind speed/force data for arbitrary time periods across different latitudinal bins.
    
    Parameters
    ---
    results: Dictionary containing processed data for each basin.
    basin_name: Name of the basin to plot (e.g., "North Atlantic").
    periods: List of period boundaries (e.g., [1830, 1836, 1850, 1860]).
    """
    #get amoount of future lines, create list of colors
    N_lines = len(periods) - 1
    start = 0.0
    stop = 1.0
    cm_subsection = np.linspace(start, stop, N_lines) 
    colors = [ cm.viridis(x) for x in cm_subsection ]
    
    # Get the data for the specific basin
    basin_data = results.get(basin_name, None)
    if basin_data is None:
        print(f"No data found for {basin_name}")
        return

    ax_.set_xlabel('wind speed [Beaufort]')
    ax_.set_title(f'whaler {basin_name}')
    #ax_.set_yticklabels([])

    # Plot each period
    for i, (start, end) in enumerate(zip(periods[:-1], periods[1:])):
        period_name = f"period_{i+1}_{start}_{end}"
        period_name_amount = f"N_period_{i+1}_{start}_{end}"
        if period_name in basin_data:
            period_data = basin_data[period_name]
            mean_speeds = period_data['speed'].values
            lats = period_data['lat_bin'].values
            amount = basin_data[period_name_amount]
            # Plot the data for this period
            ax_.plot(mean_speeds, lats, label=f'{start}-{end-1}, N= {amount}', color = colors[i])#, marker='o'
        else:
            print(f"Warning: No data for {period_name}")
    
    # Add legend and grid
    speed_all = basin_data['all']['speed'].values
    lat_all = basin_data['all']['lat_bin'].values
    amount_all = basin_data['N_total']
    ax_.plot(speed_all, lat_all, label=f'all, N= {amount_all}',color = 'black', linestyle = '--')#, marker='o'
    ax_.legend(title="Time Periods")

    
def plot_20CR_data(results, basin_name, ax_, startyear, endyear):
    """
    Plot the wind speed/force data for arbitrary time periods across different latitudinal bins.
    
    Parameters
    ---
    results: Dictionary containing processed data for each basin.
    basin_name: Name of the basin to plot (e.g., "North Atlantic").
    
    """
   
    # Get the data for the specific basin
    basin_data = results.get(basin_name, None)
    if basin_data is None:
        print(f"No data found for {basin_name}")
        return

    basin_data = basin_data[(basin_data['year'] >= startyear) & (basin_data['year'] <= endyear) ]
    ax_.set_xlabel('wind speed [Beaufort]')
    ax_.set_title(f'20CR {basin_name}')
    #ax_.set_yticklabels([])
    start = 0.0
    stop = 1.0
    number_of_lines= len(basin_data['year'].unique())
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    colors = [ cm.viridis(x) for x in cm_subsection ]

    for i, k in zip(basin_data['year'].unique(), colors):
        z = basin_data[basin_data['year'] == i]
        ax_.plot(z['speed'], z['Latitude_bin'], color = k)
        
def plot_20CR_data_trend(results, basin_name, ax_, periods, basins, binstep):
    """
    Plot the zonal mean wind speed/force data for the mean of an arbitrary time period across different latitudinal bins.
    
    Parameters
    ---
    results: Dictionary containing processed data for each basin.
    basin_name: Name of the basin to plot (e.g., "North Atlantic").
    
    """
   
    # Get the data for the specific basin
    basin_data = results.get(basin_name, None)
    latmin, latmax = basins[basin_name][0:2]
    bins = int((latmax - latmin)/binstep)
    if basin_data is None:
        print(f"No data found for {basin_name}")
        return

    basin_data = basin_data[(basin_data['year'] >= periods[0]) & (basin_data['year'] <= periods[-1]) ]
    period_data = {}
    for i, (start, end) in enumerate(zip(periods[:-1], periods[1:])):
        period_name = f"period_{i+1}_{start}_{end}"
        period_df = basin_data[(basin_data['year'] >= start) & (basin_data['year'] < end)]
        period_df = period_df.groupby(pd.cut(period_df['Latitude_bin'], bins)).mean().rename(columns={'Latitude_bin': "lat_mean"}).reset_index()

        #period_df['speed'] = period_df['speed']

        period_data[period_name] = period_df
    
    start = 0.0
    stop = 1.0
    cm_subsection = np.linspace(start, stop, len(periods[0:5])-1) 
    colors = [ cm.viridis(x) for x in cm_subsection ]
    start_sub = 0.4
    stop_sub = 1.0
    cm_subsubsection = np.linspace(start_sub, stop_sub, len(periods) - len(periods[0:5]))
    sub_colors = [ cm.Reds(x) for x in cm_subsubsection ]
    for i in range(0,len(sub_colors)):
        colors.append(sub_colors[i])
    
    
    ax_.set_xlim((-6, 3))

    ax_.tick_params(axis='x', colors='maroon') 
    ax_.xaxis.label.set_color('maroon')        

    ax_.grid(False)

    for i, (start, end) in enumerate(zip(periods[:-1], periods[1:])):
        period_name = f"period_{i+1}_{start}_{end}"
        period = period_data[period_name]
        mean_speeds = period['speed'].values
        lats = period['lat_mean'].values
        ax_.plot(mean_speeds, lats, label=f'{start}-{end-1}', color=colors[i])

    legend = ax_.legend()
    legend.get_frame().set_edgecolor('maroon') 

    ax_.set_xlabel('wind speed [Beaufort] 30-yr periods')

def plot_amount_per_bin(results, basin_name, ax_):
    """
    plot the number of observations per bin
    """
    basin_data = results.get(basin_name, None)    
    ax_.barh(basin_data['bins']['lat_bin'], basin_data['bins']['amount'],color='#434C5E', height =2, alpha=0.8)
    ax_.set(xlabel = 'Number', ylabel = 'Latitude', title = 'Number of\nObservations')