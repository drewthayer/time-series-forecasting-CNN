import pandas as pd

def downsample_df(df, time_alias):
    ''' daily: 'D' '''
    group = df.resample(time_alias)
    df_resampled = group.sum()
    return df_resampled
