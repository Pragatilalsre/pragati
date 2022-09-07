import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
register_matplotlib_converters()


def reading_file():
    sales = pd.read_csv(r'D:\pythonProject\Time-Series-Analysis\perrin-freres-monthly-champagne-.csv')
    return sales


def cleaning_data():
    sales = reading_file()
    sales.columns = ["Month", "Sales"]
    sales.drop(106, axis=0, inplace=True)
    sales.drop(105, axis=0, inplace=True)
    sales['Month'] = pd.to_datetime(sales['Month'])
    sales.set_index('Month', inplace=True)
    return sales

def select_data():
    print("Select Start_year and End_year between 1964 to 1972")
    start_year = int(input("start year"))
    end_year = int(input("end year"))
    if start_year < end_year:
        print("select spliting percentage")
        n = float(input()) / 100
        print("Select MOdel ARIMA OR SARIMAX")
        model = input()
        if (model == "ARIMA"):
            arimamodel(start_year, end_year, n)
        elif (model == "SARIMAX"):
            sarimaxmodel(start_year, end_year, n)
        elif (model == "ETS"):
            etsmodel(start_year, end_year, n)
        else:
            print("Please enter correct model")

    else:
        print("enter correct value")


def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF TEST Statistics', 'P Value', 'Lags used', 'Number of observation used']
    for value, label in zip(result, labels):
        print(label + ':' + str(value))
    if result[1] <= 0.05:
        print("data is stationary")
    else:
        print("Data is not Stationary")


def differencing():
    sales = cleaning_data()
    sales["Seasonal Diffrenece"] = sales['Sales'] - sales['Sales'].shift(12)
    adfuller_test(sales['Seasonal Diffrenece'].dropna())
    return sales


def arimamodel(start_year, end_year, n):
    if (start_year < end_year):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        Sales = cleaning_data()
        Sales = Sales[start_date:end_date]
        train_size = int(len(Sales) * (1 - n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        arima_model = sm.tsa.arima.ARIMA(train['Sales'], order=(1, 1, 1))
        arima_model_fit = arima_model.fit()
        train['forecast'] = arima_model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=True)
        residuals_train = train['Sales'] - train['forecast']
        print('Mean Absolute Percent Error For Train data:', round(np.mean(abs(residuals_train / train['Sales'])), 4))
        print('Root Mean Squared Error for Train data:', np.sqrt(np.mean(residuals_train ** 2)))
        test['forecast'] = arima_model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=True)
        residuals_test = test['Sales'] - test['forecast']
        print('Mean Absolute Percent Error for Test data:', round(np.mean(abs(residuals_test / test['Sales'])), 4))
        print('Root Mean Squared Error for Test data:', np.sqrt(np.mean(residuals_test ** 2)))
        arimamodel(start_year + 1, end_year + 1, n)


def sarimaxmodel(start_year, end_year, n):
    if (start_year < end_year):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        Sales = cleaning_data()
        Sales = Sales[start_date:end_date]
        train_size = int(len(Sales) * (1 - n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        sarimax_model = SARIMAX(train['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        Sarimax_model_fit = sarimax_model.fit()
        train['forecast'] = Sarimax_model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=True)
        residuals_train = train['Sales'] - train['forecast']
        print('Mean Absolute Percent Error For Train data:', round(np.mean(abs(residuals_train / train['Sales'])), 4))
        print('Root Mean Squared Error for Train data:', np.sqrt(np.mean(residuals_train ** 2)))
        test['forecast'] = Sarimax_model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=True)
        residuals_test = test['Sales'] - test['forecast']
        print('Mean Absolute Percent Error for Test data:', round(np.mean(abs(residuals_test / test['Sales'])), 4))
        print('Root Mean Squared Error for Test data:', np.sqrt(np.mean(residuals_test ** 2)))
        sarimaxmodel(start_year + 1, end_year + 1, n)



def etsmodel(start_year, end_year, n):
    if (start_year < end_year):
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        Sales = cleaning_data()
        Sales = Sales[start_date:end_date]
        train_size = int(len(Sales) * (1 - n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        ets_model = ETSModel(train['Sales'], error="add", trend="add", seasonal="add", damped_trend=True, seasonal_periods=12)
        ets_model_fit = ets_model.fit()
        train['forecast'] = ets_model_fit.predict(start=train.index[0], end=train.index[-1], dynamic=True)
        residuals_train = train['Sales'] - train['forecast']
        print('Mean Absolute Percent Error For Train data:', round(np.mean(abs(residuals_train / train['Sales'])), 4))
        print('Root Mean Squared Error for Train data:', np.sqrt(np.mean(residuals_train ** 2)))
        test['forecast'] = ets_model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=True)
        residuals_test = test['Sales'] - test['forecast']
        print('Mean Absolute Percent Error for Test data:', round(np.mean(abs(residuals_test / test['Sales'])), 4))
        print('Root Mean Squared Error for Test data:', np.sqrt(np.mean(residuals_test ** 2)))
        etsmodel(start_year, end_year, n)


if __name__ == "__main__":
    select_data()

