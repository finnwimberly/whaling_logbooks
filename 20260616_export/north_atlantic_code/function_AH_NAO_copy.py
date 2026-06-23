import numpy as np
import xarray as xr
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

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

def up(data, season, detrend, bounds):
    """
    Returns data with AH, IL, and NA according to the season and creates similar lat/lon values.
    
    Parameters:
    - data: xarray Dataset containing the data
    - season: string indicating the season ('winter', 'spring', 'summer', 'fall', 'all')
    - detrend: boolean indicating whether to detrend the data
    - bounds: dictionary containing the latitude and longitude bounds for different regions
    
    Returns:
    - AH: data for Azores High
    - IL: data for Iceland Low
    - NA: data for North Atlantic
    """
    latmin_NA, latmax_NA, lonmin_NA, lonmax_NA = bounds["North Atlantic"][0:4]
    latmin_AH, latmax_AH, lonmin_AH, lonmax_AH = bounds["Azores High"][0:4]
    latmin_IL, latmax_IL, lonmin_IL, lonmax_IL = bounds["Iceland Low"][0:4]

    # Sort data by latitude
    data = data.sortby('lat', ascending=True)

    # Adjust longitude values if necessary
    if np.max(data.lon) >= 181:
        data = data.assign_coords(lon=(((data.lon + 180) % 360) - 180)).sortby('lon', ascending=True)

    # Adjust prmsl values if necessary
    if np.max(data['prmsl']) >= 10000:
        data['prmsl'] = data['prmsl'] / 100

    # Group data by season
    datas = data.groupby('time.season')
    dss = datas.get(season.upper(), data) if season != 'all' else data

    if detrend:
        dss = detrend_dim(dss, dss, 'time')

    # Select regions based on latitude and longitude bounds
    IL = dss.sel(lat=slice(latmin_IL, latmax_IL), lon=slice(lonmin_IL, lonmax_IL))
    AH = dss.sel(lat=slice(latmin_AH, latmax_AH), lon=slice(lonmin_AH, lonmax_AH))
    NA = dss.sel(lat=slice(latmin_NA, latmax_NA), lon=slice(lonmin_NA, lonmax_NA))
        
    return AH, IL, NA

def area_grid(data, lat, lon):
    """
    Calculate the grid area in km^2.
    
    Parameters
    ---
    lat, lon : array of lat and lon values
    data : xarray dataset containing the data
    
    Returns
    ---
    grid area in km^2
    
    """
    grid = xr.zeros_like(data)
    for i in range(grid.shape[1]):
        grid[:, i, :] = (np.cos(np.radians(np.abs(lat[i])))*abs(lon[0]-lon[1])*111.322)*(111*np.abs(lat[1]-lat[0]))
    gridn = grid.rename('km2')
    return gridn

def area_sum_thresh(data, grid_area):
    """
    Sum the area where the condition is met.
    
    Parameters
    ---
    data: xarray dataset containing 1 where area is true
    grid_ara: output of function area_grid
    
    Returns:
    time series of sum of the area where the condition is met
    """
    relevant = xr.where(data == 1, grid_area, np.nan)
    sum_series = relevant.sum(['lat', 'lon'], skipna=True)
    return sum_series

def std_season_sort(datawhole, data, average, DJF = False):  
    """
    Returns timeseries of area of the Azores High for all seasons but DJF 
    
    Parameters
    ---
    datawhole : input data, xarray
    data : input data, xarray
    average : amount of days time series should be averaged over
    
    Returns:
    Returns timeseries of area of the Azores High for all seasons but DJF 
    
    """
    ds_ymean = datawhole.groupby('time.year').mean()        
    ds_std = ds_ymean.std(dim=['year', 'lat', 'lon'])

    data_seas = data.groupby('time.season')
    season_means = {
        'MAM': data_seas['MAM'].mean(),
        'JJA': data_seas['JJA'].mean(),
        'SON': data_seas['SON'].mean(),
        'DJF': data_seas['DJF'].mean()
    }

    
    thresholds = {
        'MAM': 0.3 * ds_std,
        'JJA': 0.5 * ds_std,
        'SON': 0.5 * ds_std,
        'DJF': 0.5 * ds_std
    }

    result = {}  


    for season, mean in season_means.items():
        thresh = thresholds[season]
        data_season = data_seas[season]['prmsl']
        ds_std_area = xr.where(data_season >= mean + thresh, data_season, np.nan)
        ds_std_01 = xr.where(data_season >= mean + thresh, 1, 0)
        grid_area = area_grid(data_season, data['lat'].values, data['lon'].values)
        time_series = area_sum_thresh(ds_std_01, grid_area)
        result[season] = time_series

    if DJF == False:
        time_s = xr.concat([result['MAM'], result['JJA'], result['SON']], dim='time').sortby('time')
    elif DJF == True:
        time_s = xr.concat([result['MAM'], result['JJA'], result['SON'], result['DJF']], dim='time').sortby('time')
        
    return time_s

def detrend_dim(globe, da, dim, deg=1):
    """
    Detrend data along a single dimension.
    
    Parameters
    ---
    globe : global xarray dataarray or dataset
    da : data that should be detrended
    dim : string, dimension of the dataset along which it should be detrended
    
    Returns
    ---
    a detrended xarray dataset / dataarray
    """
    p = globe.mean(['lat', 'lon']).polyfit(dim=dim, deg=deg, skipna=True)
    fit = xr.polyval(da.mean(['lat', 'lon'])[dim], p.prmsl_polyfit_coefficients)
    zero = fit - fit[0]
    return da - zero

def amount_mask_quantile(mini, maxi, da, year, bigsmall):
    """
    Mask data based on quantile thresholds and calculate rolling sum.
    
    Parameters
    ---
    mini, maxi : quantiles of the data you are interested in
    data : data, xarray dataset or dataarray
    year: amount of years you want to take a running mean over
    bigsmall: string, if set to 'big' you are interested in the extremely big events
    
    Returns
    ---
    quantiles, amount of events in running mean, mask
    """
    qmi, qma = da['prmsl'].quantile([mini, maxi])
    mask = xr.where(da['prmsl'] >= qma, da, np.nan) if bigsmall == 'big' else xr.where(da['prmsl'] <= qmi, da, np.nan)
    amount_roll = mask.groupby('time.year').count().rolling(year=year, center=False).sum()
    return qmi, qma, amount_roll, mask

def extreme(data1, data2, ILAH, bigsmall1, bigsmall2,smallthresh, bigthresh):
    """
    Identifies extreme events based on quantile thresholds for two datasets.

    Parameters
    ---
    data1: xarray Dataset, first dataset; for example AH
    data2: xarray Dataset, second dataset; for example AH
    ILAH: xarray DataArray or Dataset, reference data, covering the reference area
    bigsmall1: str, 'big' or 'small' for the first dataset.
    bigsmall2: str, 'big' or 'small' for the second dataset.

    Returns:
    xarray DataArray or Dataset with extreme events matching both datasets.
    """

    def get_quantiles_and_masks(data, size):
        return amount_mask_quantile(smallthresh, bigthresh, data, 25, size)

    qm1, q1, great_events_roll_1, mask_1 = get_quantiles_and_masks(data1, bigsmall1)
    qm2, q2, great_events_roll_2, mask_2 = get_quantiles_and_masks(data2, bigsmall2)

    def get_ex_na(mask):
        ex = xr.where(np.isnan(mask), np.nan, ILAH)
        return ex.dropna('time')

    ex1na = get_ex_na(mask_1)
    ex2na = get_ex_na(mask_2)

    ex1ex2 = xr.where(ex1na['time'].isin(ex2na['time']), ex1na, np.nan)
    ex1ex2na = ex1ex2.dropna('time')

    return ex1ex2na

def categorize_ah_series(ah_series, ah_ex, ah_exm):
    """
    Categorize AH series based on the presence of extreme values.

    Parameters
    ---
    ah_series: xarray DataArray, the AH series to categorize.
    ah_ex: xarray DataArray, extreme big AH values.
    ah_exm: xarray DataArray, extreme small AH values.

    Returns:
    AH_big: xarray DataArray, categorized AH series.
    """

    ah_big = xr.where(ah_series.time.isin(ah_ex.time), 10,
                      xr.where(ah_series.time.isin(ah_exm.time), -10, 0))
    return ah_big

def filter_by_category(ah_big_d, category_value):
    """
    Filter AH data based on a specific category value.

    Parameters
    ---
    ah_big_d: xarray DataArray, resampled AH data with number indicating if big small or neutral
    category_value: int, value to filter for:
        10: big
        -10: small
        0: neutral

    Returns
    ---
    Filtered xarray DataArray.
    """
    return xr.where(ah_big_d == category_value, ah_big_d, np.nan).dropna('time')

def get_whaling_data(na_whaling, category_data):
    """
    Extract NA whaling data based on the presence of specific AH categories.

    Parameters
    ---
    na_whaling: xarray DataArray, NA whaling data.
    category_data: xarray DataArray, data representing specific AH categories.

    Returns
    ---
    Filtered xarray DataArray.
    """
    return xr.where(na_whaling.time.dt.date.isin(category_data.time.dt.date), na_whaling, np.nan).dropna('time', 'all')

def adjust_wind_components(data):
    """
    Make westerly winds positive, easterly winds negative.
    
    Parameters
    ---
    data: xarray dataarray
    
    Returns:
    adjusted xarray dataarray
    """
    # Adjust uwnd based on wind angle
    data['uwndB'] = xr.where(data['angle'] <= 180, data['speed'] * -1, data['speed'])
    
    # Adjust vwnd based on wind angle
    data['vwndB'] = xr.where((data['angle'] > 270) | (data['angle'] < 45), data['speed'] * -1, data['speed'])
    
    # Invert vwndB values for correct orientation
    data['vwndB'] = data['vwndB'] * -1
    
    return data

# Preprocess NAO data
def preprocess_nao(NAO):
    """
    prepare the NAO index
    
    Parameters
    ---
    NAO: NAO index
    
    Returns
    --- 
    dataframe of scaled NAO index
    """
    NAOf = np.where(NAO == -99.99, np.nan, NAO)
    NAOdf = pd.DataFrame(data=NAOf, columns=['year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    NAOdf['NAO index'] = NAOdf.loc[:, 'Jan':'Dec'].mean(axis=1, skipna=True)
    NAO_std = np.std(NAOdf['NAO index'])
    NAOdf['scale'] = NAOdf['NAO index'] / NAO_std
    return NAOdf

def create_nao_xarray(NAOdf, start_date='1/1/1836', end_date='12/1/2015'):
    """ 
    create xarray dataarray from pandas dataframe between start and end date
    
    Parameters
    ---
    NAOdf: pandas dataframe of NAO index
    
    Returns
    ---
    xarray dataarray of NAO index
    """
    NAO_times = pd.date_range(start=start_date, end=end_date, freq='MS')
    NAO_xr = xr.DataArray(
        data=NAOdf.loc[:, 'Jan':'Dec'].to_numpy().ravel(),
        dims=["time"],
        coords={"time": NAO_times}
    )
    return NAO_xr

# Determine NAO conditions based on season
def analyze_nao(NAOdf, whaling_xr, winter):
    """
    sort whaling data by NAO
    
    Parameters
    ---
    NAOdf: pandas dataframe of NAO index
    whaling_xr: xarray dataset of whaling data
    winter: if winter == False, winter (DJF) is excluded from analysis
    
    Returns:
    xarray dataarray of NAO index
    """
    NAO_xr = create_nao_xarray(NAOdf)
    NAO_xr = NAO_xr.sel(time=slice(np.min(whaling_xr.time), np.max(whaling_xr.time)))
    
    if winter:
        NAO_xr = NAO_xr.sel(time=NAO_xr.time.dt.month.isin([3,4,5,6,7,8,9,10,11]))
    else:
        NAO_xr = NAO_xr

    NAOscale = NAO_xr / NAO_xr.std()
    NAOscale_d = NAOscale.resample(time='D').nearest(tolerance="30D")

    NAOscale_d = NAOscale_d
    q10, q90 = NAOscale_d.quantile([0.1, 0.9])
    NAOpos = NAOscale_d.where(NAOscale_d > q90).dropna('time')
    NAOneg = NAOscale_d.where(NAOscale_d < q10).dropna('time')
    NAOneut = NAOscale_d.where((NAOscale_d <= q90) & (NAOscale_d >= q10)).dropna('time')
    
    return NAOpos, NAOneg, NAOneut


def process_nao(NAO, whaling_xr, winter):
    """ 
    create xarray dataarray of positive, negative and neutral NAO
    
    Parameters
    ---
    NAO: original NAO index
    whaling_xr: xarrray dataarray of data
    winter: boolean, decide if winter should be excluded from analysis
    
    Returns
    ---
    xarray dataarrays of NAO index according to its state
    """
    NAOdf = preprocess_nao(NAO)
    NAOpos, NAOneg, NAOneut = analyze_nao(NAOdf, whaling_xr, winter)
    return NAOpos, NAOneg, NAOneut

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

##
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
    open reanalysis data in certain area
    
    Parameters
    ---
    datau: zonal wind speed [m/s]
    datav: meridional wind speed [m/s]
    latmin, latmax, lonmin, lonmax: lat and lon bounds of desired region
    
    Returns
    ---
    xarray dataset of windspeeds in Beaufort and corresponing wind directions in degrees
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

def MC_bootstrap(to_test, data_all, test_n, maps):
    """
    Monte Carlo Bootstrap test
    
    Parameters
    ---
    to_test: xarray dataset of data to test
    data_all: the available data
    test_n: amount of iterations
    maps: if True: data for visualizing it on a map
    
    Returns
    ---
    quantiles for 1% and 99% confidence intervals, distribution of test
    """
    chunksize = int(test_n / 4)
    chunks = int(test_n/chunksize)

    times = data_all.time.values
    iterations = len(to_test.time)

    if maps:
        helping = to_test.drop('time').mean('time')
    else:
        helping = to_test.mean(['lon']).drop('time').mean('time')

    distr_mean_h = xr.zeros_like(helping)
    n_n = np.arange(iterations)
    distr_mean_final = distr_mean_h.expand_dims(dim={"distr": [0]})

    dist_array = distr_mean_h.expand_dims(dim={"distr": iterations}).assign_coords(distr=("distr", n_n)).copy()

    for k in range(chunks):
        xmin = k * chunksize
        xmax = xmin + chunksize
        iteration_n = np.arange(xmin, xmax)
        distr_mean = distr_mean_h.expand_dims(dim={"distr": iteration_n}).copy()

        random_times = np.random.choice(times, (chunksize, iterations))

        for i in range(chunksize):
            temp_array = np.array([data_all.sel(time=t).drop('time').values for t in random_times[i]])
            temp_dataarray = xr.DataArray(temp_array, dims=dist_array.dims, coords=dist_array.coords).copy()
            dist_array.loc[{'distr': n_n}] = temp_dataarray
            dist_m = dist_array.mean('distr')
            distr_mean.loc[{'distr': xmin + i}] = dist_m
            #print(f"{k} file {i}")

        distr_mean_final = xr.concat([distr_mean_final, distr_mean], dim='distr')

    distr_mean_final = distr_mean_final.sel(distr=slice(0, test_n - 1))
    quans = distr_mean_final.quantile(0.01, dim='distr')
    quanb = distr_mean_final.quantile(0.99, dim='distr')

    return quans, quanb, distr_mean_final


def plot_data(whaling_data, model_data, whaling_bin_data, df_data,NA_whaling, quans_data, quanb_data, minneut_data, title, color, ax0_title):
    sns.set(style='whitegrid', context='poster')
    
    fig = plt.figure(figsize=(43, 14), constrained_layout=True)
    gs = GridSpec(4, 12, figure=fig)
    
    ax0 = fig.add_subplot(gs[:, 0:6], projection=ccrs.PlateCarree())
    ax05 = fig.add_subplot(gs[:, 6:7])
    ax1 = fig.add_subplot(gs[:, 7:9])
    ax2 = fig.add_subplot(gs[:, 9:11], sharex=ax1)
    
    # Plot whaling data
    whaling_data['uwnd'].mean(['time', 'lon']).groupby_bins('lat', 25).mean().plot(y='lat_bins', label=title, color=color, ax=ax1)
    NA_whaling['uwnd'].mean(['time', 'lon']).groupby_bins('lat', 25).mean().plot(y='lat_bins', label='whaler mean', ax=ax1,color='#C18C21')
    
    # Plot model data
    model_data['uwndB'].mean(['time', 'lon']).plot(y='lat', ax=ax2, color=color, label=f'20CR {title}')
    data_model['uwndB'].mean(['time', 'lon']).plot(y='lat', ax=ax2, color='#C18C21', label='20CR mean')
    ax2.fill_betweenx(quans_data.lat.values, quans_data.values, quanb_data.values, color='silver', alpha=0.7)
    
    # Plot number of observations
    ax05.barh(whaling_bin_data['lat_bin'], whaling_bin_data['uwnd'], color='#434C5E', height=2, alpha=0.8)
    sns.despine(ax=ax05)
    
    # Plot contour map
    levels = np.arange(-1, 1.05, 0.05)
    im = minneut_data.plot.contourf(ax=ax0, cmap='RdBu_r', add_colorbar=False, levels=levels)
    ax0.coastlines()
    ax0.set_extent([-90, 0, 0, 70], crs=ccrs.PlateCarree())
    ax0.add_feature(cf.NaturalEarthFeature('physical', 'land', '50m', facecolor='snow', edgecolor='k', linewidth=1.0))
    ax0.set_title(ax0_title)
    
    gl = ax0.gridlines(color='lightgrey', linestyle='-', draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    sns.scatterplot(x=df_data['lon'], y=df_data['lat'], hue=df_data['uwnd'], palette='PRGn_r', ax=ax0)
    
    fig.subplots_adjust(right=0.835)
    cbar_ax = fig.add_axes([0.1, 0.1, .01, .8])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.set_label('absolute speed anomaly [Beaufort]', rotation=90)
    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')
    
    ax0_ylim = ax0.get_ylim()
    ax05.set_ylim(ax0_ylim)
    for ax_ in [ax1, ax2]:
        ax_.set_ylim(ax0_ylim)
        ax_.set_yticklabels([])
        ax_.set(ylabel=None, xlabel='Beaufort')
        legend = ax_.legend(loc='upper right')
        frame = legend.get_frame()
        sns.despine(ax=ax_)
    
    ax05.set(ylabel='°N', xlabel='Number')
    ax05.set_title('Number of\nObservations')
    ax1.set_title('whaler')
    ax2.set_title('20cr')
    
    plt.show()