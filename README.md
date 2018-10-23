# time-series-forecasting-CNN
This is my work following a tutorial on using a convolutional neural net for time series forecasting. The tutorial provides a dataset and examples of engineering the data and implementing the modeling with Keras.

tutorial: https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/

Here, I follow the tutorial's examples and factor the logic into modules and functions, the way I would use it in production.

__Business problem:__
Given some number of prior days of total daily power consumption, predict the next standard week of daily power consumption.

__Data:__ 'Household Power Consumption' dataset from UCI machine learning repository
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
__train_test_split.py__
  - train 2006:2009
  - validate 2010
  - data in standard weeks (saturday to sunday) for easy interpretation
  - training set duration: 1113 days, 159.0 weeks
  - test set duration: 322 days, 46.0 weeks

### time series forecast training data
  - training features in sets of 7-day intervals
  - training targets in sets of 7-day intervals, offset by 7 days (so X_train[7,:] = y_train[0,:], and so on...)

### training: univariate
__train_model.py__
  - training on 'global_active_power', the total power used by the house

### model performance
__evaluate_model.py__

preliminary result:
<img alt="output 1" src="/figures/output_1_predictions.png" width='500'>

rmse error:
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
