# time-series-forecasting-CNN
Using a convolutional neural net for time series forecasting.

Given some number of prior days of total daily power consumption, predict the next standard week of daily power consumption.

Data: kilowatts
  - units: kilowatts
  - time range: 2006 to 2010

Forecast modeling:
  - forecast will be days 1 to 7 into the future

Model evaluation:
  - evaluate each forecast day individually
  - use RMSE so we can get an error metric in the data units (kilowatts)

### prepare data  
__load_and_clean_data.py__
  - pandas dataframe
  - set datetime indices
  - impute missing values
  - feature engineering

__downsample_data.py__
  - downsample to daily sums
  - pandas offset aliases: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

### prepare training and validations sets
  - train 2006:2009
  - validate 2010
  - data in standard weeks (saturday to sunday) for easy interpretation
  - training set duration: 1113 days, 159.0 weeks
  - test set duration: 322 days, 46.0 weeks
