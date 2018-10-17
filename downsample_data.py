import pandas as pd
import pickle
import os

from DataTools.resample import downsample_df
from DataTools.pickle import save_to_pickle

if __name__=='__main__':
    # LOAD DATA
    dir = 'data'
    fname = 'data.pkl'
    with open(os.path.join(dir,fname), 'rb') as f:
        data = pickle.load(f)

    data_resamp = downsample_df(data, 'D')

    # SAVE DATA 
    save_to_pickle(data_resamp, dir, 'data_daily.pkl')
