import pandas as pd
import numpy as np
import xarray as xr
import pickle

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
                   #"Entry Date Time" : "DateTime", 
                   "Wind Direction":"Wind Direction description","WD_Bearing" : "Wind Direction" })
    
    # Add year, month, and date columns
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
    #only between 1820 and 1900
    datana = datana[(datana['year'] >= 1820) & (datana['year'] < 1900)]
    return datana

def assign_amount_season(data):
    """
    count the amount of logbook entries per season
    
    Parameters
    ---
    data: pandas dataframe of whaling data
    
    Returns
    ---
    pandas dataframe
    """
    N_season = {season: (data['season'] == season).sum() for season in data['season'].unique()}
    data['season'] = data['season'].map(lambda season: f"{season}, N={N_season.get(season, 0)}")
    
    return data

def corr_datapoints_mw(whaler, model):
    """
    extract reanalysis data that have a corresponding logbook entry
    
    Parameters
    ---
    whaler: pandas df of whaling data
    model: reanalysis data
    
    Returns
    ---
    xarray dataarray
    """
    
    lats = xr.DataArray(whaler['Latitudegrid'], dims='z')
    lons = xr.DataArray(whaler['Longitudegrid'], dims='z')
    times = xr.DataArray(whaler['date'], dims='z')

    model_speed = model['speed'].sel(lat=lats, lon=lons, time=times, method='nearest').compute()
    model_angle = model['angle'].sel(lat=lats, lon=lons, time=times, method='nearest').compute()
    
    return model_speed, model_angle

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

def process_basin(whaler_data, model_data, basin_bounds):
    """
    create pandas df of reanalysis wind speeds and directions that have corresponding logbook entries in certain location
    
    Parameters
    ---
    whaler_data: pandas df of whaling data
    model_data: xarray dataarray of reanalysis data
    basin bounds: [latmin, latmax, lonmin, lonmax]
    
    Returns
    ---
    pandas df
    """
    latmin, latmax, lonmin, lonmax = basin_bounds
    daily_speed, daily_angle = corr_datapoints_mw(whaler_data, model_data)
    speed_df = xrtodf(daily_speed)
    angle_df = xrtodf(daily_angle)
    combined_df = pd.concat([speed_df, angle_df['angle']], axis=1).drop(columns='z')
    combined_df = combined_df.rename(columns={'speed': "Wind Speed/Force", 'angle': 'Wind Direction'})
    combined_df['month'] = combined_df['time'].dt.month
    combined_df.loc[:, 'season'] = combined_df['month'].apply(month_to_season)
    #Concatenate data based on season
    combined_df = combined_df.reset_index().rename(columns={'index': 'original_index'})
    return combined_df

def month_to_season(month):
    """
    convert month to season --> MAM, JJA, SON, DJF
    
    Parameters
    ---
    month: month
    
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

def grid_w_daily_diff(steplon, steplat, bounds, data, lon = 'Longitudegrid', lat ='Latitudegrid'):
    """
    create subgrids (smaller boxes than the original bounds) to create windroses for those
    
    Parameters
    ---
    steplon, steplat: length of box in lon, lat direction
    bounds: bounds for desired region,[latmin, latmax, lonmin, lonmax] 
    data: pandas df
    
    Returns
    ---
    grid
    """
    latmin, latmax,lonmin, lonmax = bounds
    lons = np.arange(lonmin, lonmax, steplon)
    lats = np.arange(latmin, latmax, steplat)
    grid = {}

    for valuelat in lats:
        for valuelon in lons:
            grid[f'l{valuelon}{valuelat}'] = data[(data[lon] >= valuelon) & 
                                                  (data[lon] < valuelon + steplon) & 
                                                  (data[lat] >= valuelat) & 
                                                  (data[lat] < valuelat + steplat)]
    return grid  

def sort_array(arr):
    '''
    sort seasons for plot
    
    Parameters
    ---
    arr: array of seasons
    
    Returns
    ---
    sorted seasons
    '''
    custom_order = ['JJA', 'SON', 'DJF', 'MAM']
    seasons = np.array([entry.split(',')[0] for entry in arr])
    sort_index = np.argsort([custom_order.index(season) for season in seasons])
    sorted_arr = arr[sort_index]
    return sorted_arr