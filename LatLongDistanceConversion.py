from math import pi, sin, cos, acos
import pandas as pd
def distance_between_two_points(lat1_str, lon1_str, lat2_str, lon2_str):
    try:
        lat1 = float(lat1_str)
        lon1 = float(lon1_str)
        lat2 = float(lat2_str)
        lon2 = float(lon2_str)
    except ValueError:
        return -1
    if lat1 == lat2 and lon1 == lon2:
        return 0
    radlat1 = pi * lat1 / 180
    radlat2 = pi * lat2 / 180
    theta = lon1 - lon2
    radtheta = pi * theta / 180
    dist = sin(radlat1) * sin(radlat2) + cos(radlat1) * cos(radlat2) * cos(radtheta)
    if dist > 1:
        dist = 1
    dist = acos(dist)
    dist = dist * 180 / pi
    return dist * 60 * 1.1515
folder = "C:\\Users\\baker.todd\\workspace\\data"
df = pd.read_csv("{}\\{}".format(folder, "ProjectedCostSQLOutput1019Thru1219NAICSCustVol.csv"))
df['DISTANCE_SHIPPER_ORIGIN'] = df.apply(lambda x: distance_between_two_points(x['SHIPPER_LATD'],
                                                                x['SHIPPER_LNGT'],
                                                                x['ORIGIN_SIC_LATD'],
                                                                x['ORIGIN_SIC_LNGT']), axis=1)
df['DISTANCE_SHIPPER_DESTINATION'] = df.apply(lambda x: distance_between_two_points(x['SHIPPER_LATD'],
                                                                x['SHIPPER_LNGT'],
                                                                x['DESTINATION_SIC_LATD'],
                                                                x['DESTINATION_SIC_LNGT']), axis=1)
df['DISTANCE_SHIPPER_CONSIGNEE'] = df.apply(lambda x: distance_between_two_points(x['SHIPPER_LATD'],
                                                                x['SHIPPER_LNGT'],
                                                                x['CONSIGNEE_LATD'],
                                                                x['CONSIGNEE_LNGT']), axis=1)
df['DISTANCE_ORIGIN_DESTINATION'] = df.apply(lambda x: distance_between_two_points(x['ORIGIN_SIC_LATD'],
                                                                x['ORIGIN_SIC_LNGT'],
                                                                x['DESTINATION_SIC_LATD'],
                                                                x['DESTINATION_SIC_LNGT']), axis=1)
df['DISTANCE_ORIGIN_CONSIGNEE'] = df.apply(lambda x: distance_between_two_points(x['ORIGIN_SIC_LATD'],
                                                                x['ORIGIN_SIC_LNGT'],
                                                                x['CONSIGNEE_LATD'],
                                                                x['CONSIGNEE_LNGT']), axis=1)
df['DISTANCE_DESTINATION_CONSIGNEE'] = df.apply(lambda x: distance_between_two_points(x['DESTINATION_SIC_LATD'],
                                                                x['DESTINATION_SIC_LNGT'],
                                                                x['CONSIGNEE_LATD'],
                                                                x['CONSIGNEE_LNGT']), axis=1)
df.to_csv('{}\\{}'.format(folder, 'ProjectedCostSQLOutputWithDistance.csv'), index=False)