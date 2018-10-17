import numpy as np
import pandas as pd

def df_impute_previous_index(df, idx_delta, nan_tag): # too slow
    ''' imputes missing values based on prior index in pandas dataframe,
        uses np array for speed. Preserves column names and indices.
        inputs:
            df:         pandas dataframe
            idx_delta:  int, number of indices to find prior value to impute
            nan_tag:    np.nan OR string, e.g. 'nan' or 'na'

        output:
            new_df: pandas dataframe, same size as df '''
    vals = df.values
    for row in range(vals.shape[0]):
        for col in range(vals.shape[1]):
            if vals[row,col] == nan_tag:
                vals[row,col] = vals[row - idx_delta, col]
    new_df = pd.DataFrame(data=vals, index=df.index, columns=df.columns)
    return new_df
