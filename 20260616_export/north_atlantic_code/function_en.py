import pandas as pd
import numpy as np
import xarray as xr
import pickle
from pathlib import Path
from typing import Optional, Dict
import string

def open_logbook(path,lonmin, lonmax, latmin, latmax, basin):
    """
    open the whaling data in the desired region
    
    Parameters
    ---
    path: file path
    lonmin, lonmax, latmin, latmax: bounds of the desired regions
    
    Returns
    ---
    pandas df
    """
     
    #with open(path,'rb') as f: data = pickle.load(f)
    data = pd.read_csv(path)
    
    # need to rename a lot because column names changed
    data = data.rename(columns={"Wind Speed/Force" :"Wind description","BF Value": "Wind Speed/Force", 
                   "Latitude" : "Latitude description", "Latitude_decimal":"Latitude",
                   "Longitude" : "Longitude description", "Longitude_decimal":"Longitude",
                  # "Entry Date Time" : "DateTime", 
                   "Wind Direction":"Wind Direction description","WD_Bearing" : "Wind Direction" })

    data['year'] = pd.DatetimeIndex(data['DateTime']).year
    data['month'] = pd.DatetimeIndex(data['DateTime']).month
    data['date'] = pd.DatetimeIndex(data['DateTime']).date
    
    if "Pacific" in basin:
        datana = data[((data['Longitude'] >= lonmax) | 
                         ((data['Longitude'] >= -180) & (data['Longitude'] <= lonmin))) & 
                        (data['Latitude'].between(latmin, latmax))]
    else:
        datana = data[(data['Longitude']>=lonmin) & (data['Longitude']<=lonmax) & (data['Latitude']>=latmin) & (data['Latitude'] <=latmax)]
    datana = datana.dropna(subset = ['DateTime', 'Wind Speed/Force', 'Wind Direction'])
    datana = datana.sort_values(by = 'date', ascending = True)

    datana['Longitudegrid'] = np.round(datana['Longitude'], decimals = 0)
    datana['Latitudegrid'] = np.round(datana['Latitude'], decimals = 0)
    datana = datana.rename(columns={"Longitude": "lon", "Latitude": "lat"})
    return datana



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

def add_time_dimension(ds: xr.Dataset, dim_name: str = 't') -> xr.Dataset:
    """
    Expand dataset to include a new time dimension.

    Parameters
    ---
    ds (xr.Dataset): Input dataset.
    dim_name (str): Name of the new dimension.

    Returns
    ---
    xr.Dataset: Expanded dataset with the new dimension.
    """
    return ds.expand_dims(dim=dim_name)

def correlate_datapoints(whaler: xr.Dataset, model: xr.Dataset) -> xr.DataArray:
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
    dates = whaler['date'].values
    lats = whaler['Latitudegrid'].values
    lons = whaler['Longitudegrid'].values
    
    lat_da = xr.DataArray(lats, dims='points')
    lon_da = xr.DataArray(lons, dims='points')
    time_da = xr.DataArray(dates, dims='points')

    model_speed = model['WSPD10m'].sel(lat=lat_da, lon=lon_da, time=time_da, method='nearest')

    return model_speed

def lon180(data):
    """ 
    sets longitude values between -180 and 180
    
    Parameters
    ---
    xarray dataset or dataarray with lat/lon as dimensions
    
    Returns
    ---
    xarray dataset or dataarray with longitude set as -180, 180
    """
    data = data.sortby('lat', ascending = True)
    if np.max(data.lon >= 181):
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon', ascending = True)
    return data

def preprocess_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess the dataset by adding a time dimension.

    Parameters
    ---
    ds: Input dataset (xr.Dataset)

    Returns
    ---
    xr.Dataset: Preprocessed dataset with time dimension.
    """
    return add_time_dimension(ds, dim_name='t')

def load_logbook(file_path: Path, lon_min: float, lon_max: float, 
                lat_min: float, lat_max: float, basin:str) -> pd.DataFrame: #xr.Dataset:
    """
    Load whaling data, masked some parts of the Pacific if basin == North Atlantic

    Parameters
    ---
    file_path: Path to the logbook file.
    lonmin, lonmax, latmin, latmax: bounds of the desired region

    Returns:
    pd.dataframe
    """
    datana = open_logbook(file_path, lon_min, lon_max, lat_min, lat_max, basin)

    if basin == 'North Atlantic':
        mask = (datana['lon'] < -70) & (datana['lat'] < 10)
        datana = datana.where(~mask)
    
    return datana

def process_year(
    year: int,
    datana: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    data_dir: Path,
    basin : str
) -> Optional[pd.DataFrame]:
    """
    Process data for every ensemble member of a single year

    Parameters
    ---
    year: Year to process.
    datana: Logbook data.
    lon_min:, lonmax, latmin, latmax: bounds of the desired region
    data_dir: Base directory for reanalysis

    Returns:
    Optional[pd.DataFrame]: DataFrame with correlated wind speeds or None if no data.
    """

    file_pattern = data_dir / f"extraction{year}_dir" / str(year) / f"WSPD10m.{year}.daily_mem*.nc"
    
    try:
        re = xr.open_mfdataset(
            str(file_pattern),
            data_vars='minimal',
            coords='minimal',
            compat='override',
            engine='netcdf4',
            concat_dim='t',
            combine='nested',
            preprocess=preprocess_dataset,
            parallel=True
        )
    except FileNotFoundError:
        print(f"Warning: Data files for year {year} not found. Skipping.")
        return None

    re = lon180(re)
    
    #### test
    if "Pacific" in basin:
        re1 = re.sel(lat=slice(lat_min, lat_max), lon=slice(lon_max, 179))
        re2 = re.sel(lat=slice(lat_min, lat_max), lon=slice(-180, lon_min))
        re_NA = xr.combine_by_coords([re1, re2])
    else:
        re_NA = re.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
    ### end test
    
    #re_NA = re.sortby('lat').sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    re_NA['WSPD10m'] = ms_bf_conv(re_NA['WSPD10m'])

    whaler_NA = datana[datana['year'] == year]

    if whaler_NA['date'].size > 0:
        wind_NA_speed = correlate_datapoints(whaler_NA, re_NA)
        return wind_NA_speed.compute().to_dataframe()
    else:
        return None


def split_dyn_periods(data_model, data_whaler, periods):
    """
    split data into desired periods
    
    Parameters
    ---
    data_model: pd.Dataframe of reanalysis data
    data_whaler: pd.Dataframe of Logbook data
    periods: array of years to split the data by
    
    Returns
    ---
    pd.Dataframe of Logbook and Reanalysis data split by the periods
    """
    period_data = {}
    
    def filter_data(df, start, end, column):
        """Helper function to filter data based on the period."""
        return df[(df['year'] >= start) & (df['year'] < end)][column].dropna()

    for i, (start, end) in enumerate(zip(periods, periods[1:])):
        period_name = f"period_{i+1}_{start}_{end-1}"

        period_df_whaler = filter_data(data_whaler, start, end, 'Wind Speed/Force')
        period_df_model = filter_data(data_model, start, end, 'WSPD10m')
        
        period_data[period_name] = pd.DataFrame({
            '20CR': period_df_model,
            'whaler': period_df_whaler
        })
        
        period_data[f"N_{period_name}"] = len(period_df_whaler)

    return period_data