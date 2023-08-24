from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET
import os
from functools import reduce


BASE_DIR = Path(__file__).resolve().parent
rawdata_filepath = str(BASE_DIR) + '/dataset/collection/'
# define location of original dataset
brent_oridata = rawdata_filepath + 'RBRTEd.xls'
wti_oridata = rawdata_filepath + 'RWTCd.xls'
fuel_oridata = rawdata_filepath + 'Fuel_Price.csv'


# convert OPEC .xml file to .csv format
def convert_xml_to_csv(oldfile, newfile):
    xmlparse = ET.parse(oldfile)
    root = xmlparse.getroot()

    list_of_rows = []
    for child in root:
        item = child.attrib
        list_of_rows.append(item)

    df = pd.DataFrame(list_of_rows)
    df.rename(columns={'data': 'Date', 'val': 'OPEC/brl'}, inplace=True)
    df.to_csv(newfile, index=False)

    return None


def convert_daily_to_weekly_average(oldfile, newfile):
    if oldfile == brent_oridata or oldfile == wti_oridata:
        df = pd.read_excel(oldfile, 'Data 1',
                           skiprows=[i for i in range(0, 2)], parse_dates=[0], index_col=0)
    else:
        df = pd.read_csv(oldfile, parse_dates=[0], index_col=0)

    if oldfile == fuel_oridata:
        df_resampled = df.resample('W', label='left').mean().round(2).bfill()
    else:
        df_resampled = df.resample('W', label='left').mean().round(2)

    df_resampled.to_csv(newfile)

    return None


def merge_files(file1, file2, file3, file4, file5, newfile):
    df1 = pd.read_table(file1, sep=',')
    df2 = pd.read_table(file2, sep=',')
    df3 = pd.read_table(file3, sep=',')
    df4 = pd.read_table(file4, sep=',')
    df5 = pd.read_table(file5, sep=',')

    data_frames = [df1, df2, df3, df4, df5]
    df_merged = reduce(lambda left, right: pd.merge(
        left, right, on=['Date']), data_frames)
    # df_merged = reduce(lambda left, right: pd.merge(left, right, on=['Day'], how='outer'), data_frames).fillna('N/A')
    df_merged.rename(columns={df_merged.columns[1]: 'RON95/litre',
                              df_merged.columns[2]: 'RON97/litre',
                              df_merged.columns[3]: 'Diesel/litre',
                              df_merged.columns[4]: 'WTI/brl',
                              df_merged.columns[5]: 'Brent/brl'}, inplace=True)
    pd.DataFrame.to_csv(df_merged, newfile, sep=',', index=False)

    return None


# Convert crude oil price from usd in barrel to myr in litre
def convert_oil_price_to_myr(oldfile, newfile):
    df = pd.read_csv(oldfile)
    # 1 barrel = 158.987295 litre
    # Crude Oil/litre in MYR = Crude Oil/158.987295 * Exchange Rates
    df[[df.columns[4], df.columns[5], df.columns[6]]] = df[[df.columns[4], df.columns[5],
                                                            df.columns[6]]].div(158.987295).multiply(df[df.columns[7]], axis='index').round(9)
    df.rename(columns={df.columns[4]: 'WTI/litre',
                       df.columns[5]: 'Brent/litre',
                       df.columns[6]: 'OPEC/litre'}, inplace=True)
    df.to_csv(newfile, index=False)


def delete_temp_files(file):
    os.remove(file) if os.path.exists(file) else print('File Not Found!')


'''
# convert opec xml file to csv
convert_xml_to_csv(opec_oridata, opec_newdata)
# convert all the dataset with daily price to weekly average
files = [[fuel_oridata, fuel_newdata],
         [brent_oridata, brent_newdata],
         [wti_oridata, wti_newdata],
         [opec_newdata, opec_newdata]]
for i in range(len(files)):
    convert_daily_to_weekly_average(files[i][0], files[i][1])
# combine all the datasets into one single dataset
merge_files(fuel_newdata, wti_newdata, brent_newdata,
            opec_newdata, forex_oridata, merged_file)
# convert crude oils with barrel and usd unit to litre and myr unit
convert_oil_price_to_myr(merged_file, final_dataset)
'''
