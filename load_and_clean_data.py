import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

from DataTools.impute import df_impute_previous_index

def main():
    # LOAD FILE
    dir = 'data'
    fname = 'household_power_consumption.txt'
    dataset = pd.read_csv(os.path.join(dir,fname), sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col='datetime')

    # CLEAN DATA
    # replace missing value with nan
    dataset.replace('?', np.nan, inplace=True) # could use from numpy import nan, nan
    # set data type as int
    dataset = dataset.astype('float32')

    # impute missing values: value of 24 hours previous
    idx_delta = 60*24 #index of 24 hours previous, timeset interval = 1 minute
    data_clean = df_impute_previous_index(dataset, idx_delta, np.nan)

    # engineer 4th sub_metering feature
    data_clean['Sub_metering_4'] = (data_clean.Global_active_power * 1000 / 60) - (data_clean.Sub_metering_1 + data_clean.Sub_metering_2 + data_clean.Sub_metering_3)

    # SAVE DATA TO PKL
    with open('data/data.pkl', 'wb') as f:
        pickle.dump(data_clean, f)
    print('data written to pkl')

if __name__=='__main__':
    main()
