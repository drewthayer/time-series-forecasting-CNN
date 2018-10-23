import numpy as np
import matplotlib.pyplot as plt

from DataTools.pickle import save_to_pickle, load_from_pickle

if __name__=='__main__':
    # load output
    (true, pred, errors) = load_from_pickle('output', 'output_1.pkl')

    # plot errors
    plt.plot(errors)
    plt.ylabel('RMSE [kW]')
    plt.title('prediction error: test period')
    plt.show()
