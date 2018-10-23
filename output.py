import numpy as np
import matplotlib.pyplot as plt
import os

from DataTools.pickle import save_to_pickle, load_from_pickle

if __name__=='__main__':
    # load output
    (true, pred, errors) = load_from_pickle('output', 'output_1.pkl')

    # plot errors
    fname = 'output_1_rmse'
    plt.plot(errors)
    plt.ylabel('RMSE [kW]')
    plt.title('prediction error: test period')
    plt.savefig(os.path.join('figures', fname + '.png'), dpi=250)
    plt.close()
