import numpy as np
import matplotlib.pyplot as plt
import os

from DataTools.pickle import save_to_pickle, load_from_pickle

def timeseries_from_staggered_timeseries_sets(data, n):
    ''' for staggered timeseries where sets are equal length, advancing by one unit '''
    # indices that contain independent sets of values
    ii = np.linspace(n, len(data), int(len(data)/ 7))
    ii = np.insert(ii, 0, 0)
    ii = np.delete(ii, -1)
    ii = ii.astype(int)

    yy = []
    for i in ii:
        yy.append(list(data[i])) # append lists
        flat = [item for sublist in yy for item in sublist]
    return flat

if __name__=='__main__':
    # load output and test set
    (true, pred, errors) = load_from_pickle('output', 'output_1.pkl')
    test_df = load_from_pickle('data', 'data_test.pkl')
    test = test_df.values

    # get single timseries for true and pred (pred is first day predition)
    n_days = 7
    yy_true = timeseries_from_staggered_timeseries_sets(true, n_days)
    yy_pred = timeseries_from_staggered_timeseries_sets(pred, n_days)

    # plot true vs predicted
    fname = 'output_1_predictions'
    plt.plot(yy_true, 'b', label='true')
    plt.plot(yy_pred, 'orange', label='predicted', linewidth=2)
    plt.ylabel('power usage [kW]')
    plt.xlabel('test period [days]')
    plt.legend()
    plt.title('first day predictions with {} day input'.format(n_days))
    plt.savefig(os.path.join('figures', fname + '.png'), dpi=250)
    plt.close()


    # plot errors
    fname = 'output_1_rmse'
    plt.plot(errors)
    plt.ylabel('RMSE [kW]')
    plt.xlabel('test period [days]')
    plt.title('prediction error: test period')
    plt.savefig(os.path.join('figures', fname + '.png'), dpi=250)
    plt.close()
