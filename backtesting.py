import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
register_matplotlib_converters()
from time import time
import xlwt
from xlwt import Workbook
import warnings
warnings.filterwarnings('ignore')

pip install xlwt
pip install openpyxl

def reading_file():
    Sales = pd.read_csv('perrin-freres-monthly-champagne-.csv')
    return Sales
  
def Cleaning_data():
    Sales= reading_file()
    Sales.columns=["Month" , "Sales"]
    Sales.drop(106 , axis=0 , inplace=True)
    Sales.drop(105 , axis=0 , inplace=True)
    Sales['Month'] = pd.to_datetime(Sales['Month'])
    Sales.set_index('Month' , inplace=True)
    return Sales
  
  
 def select_data():
    print("Select Start_year and End_year between 1964 to 1972")
    start_year= int(input())
    end_year = int(input())
    if(start_year < end_year):
        print("select spliting percentage")
        n= float(input())/100
        print("Select MOdel ARIMA OR SARIMAX")
        model=input()
        if(model=="ARIMA"):
            ArimaModel(start_year , end_year , n)
        elif(model == "SARIMAX"):
            sarimaxModel(start_year , end_year , n)
        elif(model=="ETS"):
            etsmodel(start_year , end_year , n)
        else:
            print("Please enter correct model")
    
    else :
        print("enter correct value")
def adfuller_test(Sales):
    result = adfuller(Sales)
    labels = ['ADF TEST Statistics' , 'P Value' ,'Lags used' ,'Number of observation used']
    for value , label in zip(result ,labels):
        print(label+ ':' +str(value))
    if(result[1] <= 0.05):
        print("data is stationary")
    else:
        print("Data is not Stationary")
      
def Differencing():
    Sales= Cleaning_data()
    Sales["Seasonal Diffrenece"] = Sales['Sales']- Sales['Sales'].shift(12)
    adfuller_test(Sales['Seasonal Diffrenece'].dropna())
    Sales['Seasonal Diffrenece'].plot()
    return Sales
  
def ACF_PACF_test():
    Sales= Differencing()
    fig = plt.figure(figsize = (12 ,8))
    ax1 = fig.add_subplot(211)
    fig = plot_acf(Sales['Seasonal Diffrenece'].iloc[13:] , lags=40 , ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = plot_pacf(Sales['Seasonal Diffrenece'].iloc[13:] , lags=40 , ax=ax2)
    
    
def ArimaModel(start_year , end_year , n):
    if(start_year < end_year and start_year <=1972 and end_year >=1964):
        start_date = datetime(start_year,1,1)
        end_date = datetime(end_year,12,31)
        Sales= Cleaning_data()
        Sales =Sales[start_date:end_date]
        train_size = int(len(Sales) * (1-n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        arima_model = sm.tsa.arima.ARIMA(train['Sales'], order=(1, 1, 1))
        arima_model_fit = arima_model.fit()
        train['forecast']=Sarimax_model_fit.predict(start=train.index[0],end=train.index[-1] ,dynamic=True)
        residuals_train= train['Sales']-train['forecast']
        print('Mean Absolute Percent Error For Train data:', round(np.mean(abs(residuals_train/train['Sales'])),4))
        print('Root Mean Squared Error for Train data:', np.sqrt(np.mean(residuals_train**2)))
        test['forecast']=Sarimax_model_fit.predict(start=test.index[0],end=test.index[-1] ,dynamic=True)
        residuals_test= test['Sales']-test['forecast']
        print('Mean Absolute Percent Error for Test data:', round(np.mean(abs(residuals_test/test['Sales'])),4))
        print('Root Mean Squared Error for Test data:', np.sqrt(np.mean(residuals_test**2)))
        ArimaModel(start_year+1 , end_year+1 ,n)
       
 MPSE_TRAIN=[]
RMSE_TRAIN =[]
MPSE_TEST=[]
RMSE_TEST =[]
def sarimaxModel(start_year , end_year ,n):
    if(start_year < end_year and start_year <=1972 and end_year >=1964):
        start_date = datetime(start_year,1,1)
        end_date = datetime(end_year,12,31)
        Sales= Cleaning_data()
        Sales =Sales[start_date:end_date]
        train_size = int(len(Sales) * (1-n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        sarimax_model=SARIMAX(train['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
        Sarimax_model_fit= sarimax_model.fit()
        train['forecast']=Sarimax_model_fit.predict(start=train.index[0],end=train.index[-1] ,dynamic=True)
        residuals_train= train['Sales']-train['forecast']
        MPSE_TRAIN.append(round(np.mean(abs(residuals_train/train['Sales'])),4))
        RMSE_TRAIN.append(np.sqrt(np.mean(residuals_train**2)))
        test['forecast']=Sarimax_model_fit.predict(start=test.index[0],end=test.index[-1] ,dynamic=True)
        residuals_test= test['Sales']-test['forecast']
        MPSE_TEST.append(round(np.mean(abs(residuals_test/test['Sales'])),4))
        RMSE_TEST.append(np.sqrt(np.mean(residuals_test**2)))
        sarimaxModel(start_year+1 , end_year+1 ,n)
   
    col1 = "MPSE_TRAIN"
    col2 = "RMSE_TRAIN"
    col3 = "MPSE_TEST"
    col4 = "RMSE_TEST"
    data = pd.DataFrame({col1:MPSE_TRAIN,col2: RMSE_TRAIN ,col3:MPSE_TEST ,col4:RMSE_TEST })
    data.to_excel (r'D:\pythonProject\Time-Series-Analysis\dataframe.xlsx', index = False, header=True)
    
def etsmodel(start_year , end_year , n):
    if(start_year < end_year):
        start_date = datetime(start_year,1,1)
        end_date = datetime(end_year,12,31)
        Sales= Cleaning_data()
        Sales =Sales[start_date:end_date]
        train_size = int(len(Sales) * (1-n))
        train, test = Sales[0:train_size], Sales[train_size:len(Sales)]
        ets_model = ETSModel(train['Sales'], error="add",trend="add", seasonal="add",damped_trend=True,seasonal_periods=12)
        ets_model_fit = ets_model.fit()
        train['forecast']=ets_model_fit.predict(start=train.index[0],end=train.index[-1] ,dynamic=True)
        residuals_train= train['Sales']-train['forecast']
        print('Mean Absolute Percent Error For Train data:', round(np.mean(abs(residuals_train/train['Sales'])),4))
        print('Root Mean Squared Error for Train data:', np.sqrt(np.mean(residuals_train**2)))
        test['forecast']=ets_model_fit.predict(start=test.index[0],end=test.index[-1] ,dynamic=True)
        residuals_test= test['Sales']-test['forecast']
        print('Mean Absolute Percent Error for Test data:', round(np.mean(abs(residuals_test/test['Sales'])),4))
        print('Root Mean Squared Error for Test data:', np.sqrt(np.mean(residuals_test**2)))
        etsmodel(start_year+1 , end_year+1 ,n)
    
select_data()

   
