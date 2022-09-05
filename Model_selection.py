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
    sales = cleaning_data()
    print("Select Start_year and End_year between 1964 to 1972")
    start_year = int(input("Start_year"))
    end_year = int(input("End_year"))
    start_month = int(input("start_month"))
    end_month= int(input("End_month"))
    if start_year < end_year:
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 31)
        print(start_date)
        print(end_date)
        sales = sales[start_date:end_date]
        return sales
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


def arimamodel():
    sales = select_data()
    print("select spliting percentage")
    n = float(input()) / 100
    train_size = int(len(sales) * (1 - n))
    train, test = sales[0:train_size], sales[train_size:len(sales)]
    arima_model = sm.tsa.arima.ARIMA(train['Sales'], order=(1, 1, 1))
    arima_model_fit = arima_model.fit()
    test['forecast'] = arima_model_fit.predict(start=test.index[0], end=test.index[0], dynamic=True)
    residuals = test['Sales'] - test['forecast']
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals / test['Sales'])), 4))
    print('Root Mean Squared Error:', np.sqrt(np.mean(residuals ** 2)))


def sarimaxmodel():
    sales = select_data()
    print("select spliting percentage")
    n = float(input()) / 100
    train_size = int(len(sales) * (1 - n))
    train, test = sales[0:train_size], sales[train_size:len(sales)]
    sarimax_model = SARIMAX(train['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    sarimax_model_fit = sarimax_model.fit()
    test['forecast'] = sarimax_model_fit.predict(start=test.index[0], end=test.index[0], dynamic=True)
    residuals = test['Sales'] - test['forecast']
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals / test['Sales'])), 4))
    print('Root Mean Squared Error:', np.sqrt(np.mean(residuals ** 2)))


def etsmodel():
    Sales = select_data()
    print("select spliting percentage")
    n = float(input()) / 100
    train_size = int(len(Sales) * (1 - n))
    train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
    ets_model = ETSModel(train['Sales'], error="add", trend="add", seasonal="add", damped_trend=True,seasonal_periods=12)
    ets_model_fit = ets_model.fit()
    test['forecast'] = ets_model_fit.predict(start=test.index[0], end=test.index[-1], dynamic=True)
    residuals = test['Sales'] - test['forecast']
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals / test['Sales'])), 4))
    print('Root Mean Squared Error:', np.sqrt(np.mean(residuals ** 2)))


def select_model():
    differencing()
    print("Select Model ARIMA OR SARIMAX OR ETS")
    model = input()
    if model == "ARIMA":
        arimamodel()
    elif model == "SARIMAX":
        sarimaxmodel()
    elif model == "ETS":
        etsmodel()
    else:
        print("Please enter correct model")


if __name__ == "__main__":
    select_model()

