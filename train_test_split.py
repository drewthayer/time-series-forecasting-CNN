import pandas as pd
import numpy as np

from DataTools.pickle import save_to_pickle, load_from_pickle

def main():
    # LOAD DATA
    dir = 'data'
    data = load_from_pickle(dir, 'data_daily.pkl')

    # SPLIT DATA
    # using standard weeks sunday - saturday
    # first sunday in dataset:
    day1_train = '2006-12-17'
    # first sunday in 2010:
    day1_test = '2010-1-3'
    # last saturday in dataset:
    day_last_test = '2010-11-20'

    data_train = data[data.index >= day1_train]
    data_train = data_train[data_train.index < day1_test]
    data_test = data[data.index >= day1_test]
    data_test = data_test[data_test.index <= day_last_test]

    # METRICS
    ndays_train = data_train.shape[0]
    ndays_test = data_test.shape[0]
    print('\ntraining set duration: {} days, {} weeks'.format(ndays_train, ndays_train/7))
    print('test set duration: {} days, {} weeks\n'.format(ndays_test, ndays_test/7))

    save_to_pickle(data_train, 'data', 'data_train.pkl')
    save_to_pickle(data_test, 'data', 'data_test.pkl')

if __name__=='__main__':
    main()
