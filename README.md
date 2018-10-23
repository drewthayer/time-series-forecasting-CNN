# time-series-forecasting-CNN
Tutorial: Using a convolutional neural net for time series forecasting.

__Business problem:__
Given some number of prior days of total daily power consumption, predict the next standard week of daily power consumption.

__Data:__
  - household power consumption
  - units: kilowatts
  - frequency: daily
  - time range: 2006 to 2010

__strategy:__
  - Time-sequence forecasting: autoregression
      - be able to predict a forecast for y number of days into the future based on x number of days up to current (e.g. predict next week from this week)
  - Convolutional Neural Network
      - low-bias model that can learn non-linear relationships
      - implemented in Keras

Model evaluation:
  - evaluate each forecast day individually
  - use RMSE, metric in the data units (kilowatts)

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

### time series forecast training data
  - training features in sets of 7-day intervals
  - training targets in sets of 7-day intervals, offset by 7 days (so X_train[7,:] = y_train[0,:], and so on...)

### training output
preliminary result is highly variable:

<img alt="rmse 1" src="/figures/output_1_rmse.png" width='500'>

### repository structure
~~~
.
├── DataTools                 tools module: impute, pickle, resample
├── README.md
├── data                      data sets
├── data_Xy                   data in feature/target sets (.pkl)
├── downsample_data.py        downsample based on time interval
├── evaluate_model.py         evaluate against test set
├── figures                   figures
├── load_and_clean_data.py    load dataset and clean
├── models                    models (.json) and weights (.h5)
├── output                    output: (true, predicted, errors) (.pkl)
├── output.py                 analyze and plot output
├── train_model.py            train CNN
└── train_test_split.py       split data into train and test sets based on date 
~~~
