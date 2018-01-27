
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from dateutil.relativedelta import *
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas import *

import pandas as pd
import matplotlib.pylab as plt
import statsmodels.api as sm

pd.set_option('display.max_rows', 10000)


def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window=24).mean()
    rolstd = timeseries.rolling(window=24).std()

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.Value.values.ravel(),1,autolag='AIC')
    #dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


#looks quite good => http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
def runAll():
    #FLAGS
    TEST_STATIONARITY = False
    TEST_ACF_PACF = False

    #Just one can be true at a time
    PREPARE_1M_FORECAST = True
    PREPARE_24H_FORECAST = False


    #CODE
    dateparse = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv('data9.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse)
    data_dropZero = data[(data.T != 0).any()] # drop zeros
    if PREPARE_1M_FORECAST:
        data_dropZero_upsample = data_dropZero.resample('D').mean()
    else:
        data_dropZero_upsample = data_dropZero.resample('H').mean()
    data_dropZero_upsample = data_dropZero_upsample.interpolate(method='linear')
    data = data_dropZero_upsample['2016':] #time starts from 2016 (due to OOM)
    ts = data[:'2016-12-31'] #we give our prediction data until 01.01.2017 (rest from 01.01 to 01.02 we use for testing the forecast)

    if TEST_STATIONARITY:
        test_stationarity(ts)

    ###########################

    if TEST_ACF_PACF:
        #Methods to make signal stationary (if needed)
        ts_first_diff = ts - ts.shift(1)
        ts_first_diff.dropna(inplace=True)

        ts_seasonal_diff = ts - ts.shift(12)
        ts_seasonal_diff.dropna(inplace=True)

        seasonal_first_diff = ts_first_diff - ts_first_diff.shift(12)
        seasonal_first_diff.dropna(inplace=True)
        ############################

        #Test if signal is stationary
        #test_stationarity(seasonal_first_diff)
        #############################

        #PLOT
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(seasonal_first_diff.iloc[13:], lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(seasonal_first_diff.iloc[13:], lags=40, ax=ax2)
        plt.show()

    if PREPARE_24H_FORECAST:
        start = datetime.strptime("2017-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        date_list = [start + relativedelta(hours=x) for x in range(0,24)]
        future = pd.DataFrame(index=date_list, columns= ts.columns)
        df = pd.concat([ts, future])
        #print(df)
        #MAKE_FORECAST
        if True:
            mod = sm.tsa.statespace.SARIMAX(df, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
            results = mod.fit()
            ts_forecast = results.predict()
            #print(ts_forecast);
            #PLOT_FORECAST
            if True:
                plt.subplot(211)
                plt.title('Full signal')
                plt.plot(data,color='red',label = 'Original')
                plt.plot(ts_forecast,color='blue',label = 'Forecast')
                plt.legend(loc='best')
                plt.subplot(212)
                plt.title('Zoomed signal (24 Hours)')
                plt.plot(data['2017-01-01':'2017-01-02'],color='red',label = 'Original')
                plt.plot(ts_forecast['2017-01-01':],color='blue',label = 'Forecast')
                plt.legend(loc='best')
                plt.show()
                print(data['2017-01-01':'2017-01-02'])
                print('################')
                print(ts_forecast['2017-01-01':])

    if PREPARE_1M_FORECAST:
        start = datetime.strptime("2017-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        date_list = [start + relativedelta(days=x) for x in range(0,30)]
        future = pd.DataFrame(index=date_list, columns= ts.columns)
        df = pd.concat([ts, future])
        #print(df)
        #MAKE_FORECAST
        if True:
            mod = sm.tsa.statespace.SARIMAX(df, trend='n', order=(0,1,0), seasonal_order=(1,1,1,12))
            results = mod.fit()
            ts_forecast = results.predict()
            #print(ts_forecast);
            #PLOT_FORECAST
            if True:
                plt.subplot(211)
                plt.title('Full signal')
                plt.plot(data,color='red',label = 'Original')
                plt.plot(ts_forecast,color='blue',label = 'Forecast')
                plt.legend(loc='best')
                plt.subplot(212)
                plt.title('Zoomed signal (30 Days)')
                plt.plot(data['2017-01-01':],color='red',label = 'Original')
                plt.plot(ts_forecast['2017-01-01':],color='blue',label = 'Forecast')
                plt.legend(loc='best')
                plt.show()



runAll()






############################USED JUST AS ARCHIVE#############################


#def miscTools:
    #Remove outliers (or spikes) => no needed if using log
    #ts_log = ts[np.abs(ts.Value-ts.Value.mean())<=(1*ts.Value.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
    #ts_log = ts[~(np.abs(ts.Value-ts.Value.mean())>(3*ts.Value.std()))] #or if you prefer the other way around

#ts_log = np.log(ts)

#works but not intersting => http://machinelearningmastery.com/persistence-time-series-forecasting-with-python/
#def runAll3():
    # Create lagged dataset
    #values = DataFrame(ts.values)
    #dataframe = concat([values.shift(1), values], axis=1)
    #dataframe.columns = ['t-1', 't+1']
    #print(dataframe.head(5))

    # split into train and test sets
    #X = dataframe.values
    #train_size = int(len(X) * 0.66)
    #train, test = X[1:train_size], X[train_size:]
    #train_X, train_y = train[:,0], train[:,1]
    #test_X, test_y = test[:,0], test[:,1]

    # persistence model
    #def model_persistence(x):
    #    return x

    # walk-forward validation
    #predictions = list()
    #for x in test_X:
    #    yhat = model_persistence(x)
    #    predictions.append(yhat)
    #test_score = mean_squared_error(test_y, predictions)
    #print('Test MSE: %.3f' % test_score)

    # plot predictions and expected results
    #plt.plot(train_y)
    #plt.plot([None for i in train_y] + [x for x in test_y])
    #plt.plot([None for i in train_y] + [x for x in predictions])
    #plt.show()

#not working well => https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
#def runAll():
    #moving_avg = pd.rolling_mean(ts_log,24)
    #ts_log_moving_avg_diff = ts_log - moving_avg
    #ts_log_moving_avg_diff.dropna(inplace=True)


    #Exponentially weighted moving average
    #expwighted_avg = pd.ewma(ts_log, halflife=24)
    #plt.plot(ts_log)
    #plt.plot(expwighted_avg, color='red')

    #Eliminating Trend and Seasonality
    #decomposition = seasonal_decompose(ts)
    #trend = decomposition.trend
    #seasonal = decomposition.seasonal
    #residual = decomposition.resid
    #ts_log_decompose = residual
    #ts_log_decompose.dropna(inplace=True)
    #test_stationarity(ts_log_decompose)

    ###FORECASTING EXPERIMENTAL###
    #lag_acf = acf(ts_log_decompose, nlags=20)
    #lag_pacf = pacf(ts_log_decompose, nlags=20, method='ols')

    ###FORECASTING###
    #lag_acf = acf(ts_log_diff, nlags=20)
    #lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF:
    #plt.subplot(121)
    #plt.plot(lag_acf)
    #plt.axhline(y=0,linestyle='--',color='gray')
    #plt.axhline(y=-1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    #plt.axhline(y=1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    #plt.title('Autocorrelation Function')

    # #Plot PACF:
    #plt.subplot(122)
    #plt.plot(lag_pacf)
    #plt.axhline(y=0,linestyle='--',color='gray')
    #plt.axhline(y=-1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    #plt.axhline(y=1.96/np.sqrt(len(ts_log_decompose)),linestyle='--',color='gray')
    #plt.title('Partial Autocorrelation Function')
    #plt.tight_layout()

    #AR Model
    #model = ARIMA(ts, order=(2, 1, 0))
    #results_AR = model.fit(disp=-1)
    #plt.plot(ts)
    #plt.plot(results_AR.fittedvalues, color='red')
    #plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts)**2))
    #MA Model
    #model = ARIMA(ts, order=(0, 1, 2))
    #results_MA = model.fit(disp=-1)
    #plt.plot(ts)
    #plt.plot(results_MA.fittedvalues, color='blue')
    #plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts)**2))


    #Combined Model
    #model = ARIMA(ts, order=(2, 1, 2))
    #results_ARIMA = model.fit(disp=-1)
    #plt.plot(ts)
    #plt.plot(results_ARIMA.fittedvalues, color='red')
    #plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts)**2))


    ###Taking it back to original scale###
    #predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    #print predictions_ARIMA_diff.head()

    #predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    #print predictions_ARIMA_diff_cumsum.head()

    #predictions_ARIMA_log = pd.Series(ts.ix[0], index=ts.index)
    #predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    #print(predictions_ARIMA_log.head())

    #predictions_ARIMA = np.exp(predictions_ARIMA_log)
    #plt.plot(ts)
    #plt.plot(predictions_ARIMA)
    #plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    #plt.show()


