import pandas as pd

def haversine(lat1, lon1, lat2, lon2):
    '''Calculate distance using the Haversine Formula'''
    import math

    # lon1, lat1 = coord1
    # lon2, lat2 = coord2

    R = 6371000  # Earth radius in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    km = (R * c) / 1000.0
    return round(km, 3)

def summarize_voyages(df):
    results = []

    # make a copy and ensure Entry Date Time is datetime
    df = df.copy()
    df['Entry Date Time'] = pd.to_datetime(df['Entry Date Time'], errors='coerce')

    for logbook_id, group in df.groupby('LogBook ID'):
        # drop rows where we couldn't parse the date
        group = group.dropna(subset=['Entry Date Time'])

        # if still not enough points, skip
        if len(group) < 2:
            continue

        # sort by datetime
        group = group.sort_values('Entry Date Time')

        # Time duration
        start_time = group['Entry Date Time'].iloc[0]
        end_time = group['Entry Date Time'].iloc[-1]
        duration_days = (end_time - start_time).days

        # Distance traveled
        coords = group[['Latitude_decimal', 'Longitude_decimal']].dropna().values
        if len(coords) > 1:
            distance_km = sum(
                haversine(coords[i][0], coords[i][1],
                          coords[i+1][0], coords[i+1][1])
                for i in range(len(coords) - 1)
            )
        else:
            distance_km = 0

        results.append({
            'LogBook ID': logbook_id,
            'Start Date': start_time,
            'End Date': end_time,
            'Duration (days)': duration_days,
            'Distance (km)': distance_km
        })

    return pd.DataFrame(results)