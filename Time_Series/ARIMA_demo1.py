# ARIMA demo:  http://www.cnblogs.com/foley/p/5582358.html
# 数据：AirPassenger.csv
# Python时间序列平稳检验--ADF检验： http://www.pengfoo.com/machine-learning/2017-01-24
'''
现在预测的是已有的数据，如何更改为预测未知数据，eg：预测1961-1965年的数据？
'''


import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA
import sys
from dateutil.relativedelta import relativedelta
from copy import deepcopy
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt


#################################################################################################################
import statsmodels
statsmodels.__version__

class arima_model:

    def __init__(self, ts, maxLag=9):
        self.data_ts = ts
        self.resid_ts = None
        self.predict_ts = None
        self.maxLag = maxLag
        self.p = maxLag
        self.q = maxLag
        self.properModel = None
        self.bic = sys.maxsize

    # 计算最优ARIMA模型，将相关结果赋给相应属性
    def get_proper_model(self):
        self._proper_model()
        self.predict_ts = deepcopy(self.properModel.predict())
        self.resid_ts = deepcopy(self.properModel.resid)

    # 对于给定范围内的p,q计算拟合得最好的arima模型，这里是对差分好的数据进行拟合，故差分恒为0
    def _proper_model(self):
        for p in np.arange(self.maxLag):
            for q in np.arange(self.maxLag):
                # print p,q,self.bic
                model = ARMA(self.data_ts, order=(p, q))
                try:
                    results_ARMA = model.fit(disp=-1, method='css')
                except:
                    continue
                bic = results_ARMA.bic
                # print 'bic:',bic,'self.bic:',self.bic
                if bic < self.bic:
                    self.p = p
                    self.q = q
                    self.properModel = results_ARMA
                    self.bic = bic
                    self.resid_ts = deepcopy(self.properModel.resid)
                    self.predict_ts = self.properModel.predict()



    # 参数确定模型
    def certain_model(self, p, q):
            model = ARMA(self.data_ts, order=(p, q))
            try:
                self.properModel = model.fit( disp=-1, method='css')
                self.p = p
                self.q = q
                self.bic = self.properModel.bic
                self.predict_ts = self.properModel.predict()
                self.resid_ts = deepcopy(self.properModel.resid)
            except:
                print('You can not fit the model with this parameter p,q, ' \
                      'please use the get_proper_model method to get the best model')

    # 预测第二日的值
    def forecast_next_day_value(self, type='day'):
        # 我修改了statsmodels包中arima_model的源代码，添加了constant属性，需要先运行forecast方法，为constant赋值
        self.properModel.forecast()
        if self.data_ts.index[-1] != self.resid_ts.index[-1]:
            raise ValueError('''The index is different in data_ts and resid_ts, please add new data to data_ts.
            If you just want to forecast the next day data without add the real next day data to data_ts,
            please run the predict method which arima_model included itself''')
        if not self.properModel:
            raise ValueError('The arima model have not computed, please run the proper_model method before')
        para = self.properModel.params

        # print self.properModel.params
        if self.p == 0:   # It will get all the value series with setting self.data_ts[-self.p:] when p is zero
            ma_value = self.resid_ts[-self.q:]
            values = ma_value.reindex(index=ma_value.index[::-1])
        elif self.q == 0:
            ar_value = self.data_ts[-self.p:]
            values = ar_value.reindex(index=ar_value.index[::-1])
        else:
            ar_value = self.data_ts[-self.p:]
            ar_value = ar_value.reindex(index=ar_value.index[::-1])
            ma_value = self.resid_ts[-self.q:]
            ma_value = ma_value.reindex(index=ma_value.index[::-1])
            values = ar_value.append(ma_value)

        predict_value = np.dot(para[1:], values) + self.properModel.constant[0]
        self._add_new_data(self.predict_ts, predict_value, type)
        return predict_value

    # 动态添加数据函数，针对索引是月份和日分别进行处理
    def _add_new_data(self, ts, dat, type='day'):
        if type == 'day':
            new_index = ts.index[-1] + relativedelta(days=1)
        elif type == 'month':
            new_index = ts.index[-1] + relativedelta(months=1)
        ts[new_index] = dat

    def add_today_data(self, dat, type='day'):
        self._add_new_data(self.data_ts, dat, type)
        if self.data_ts.index[-1] != self.predict_ts.index[-1]:
            raise ValueError('You must use the forecast_next_day_value method forecast the value of today before')
        self._add_new_data(self.resid_ts, self.data_ts[-1] - self.predict_ts[-1], type)

#################################################################################################################


# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('AirPassengers.csv', encoding='utf-8', index_col='Month')
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
ts = df['#Passengers']  # 生成pd.Series对象, class 'pandas.core.series.Series'

# print(ts['1949-01-01'])
# print(ts[datetime(1949, 1, 1)])
# print(ts['1949'])
# 注意时间索引的切片操作起点和尾部都是包含的，这点与数值索引的切片有所不同
# print(ts['1949-1':'1950-9'])

# 平稳性检验

# 移动平均图,size 为(加权)移动平均的时期个数
def draw_trend(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = timeSeries.ewm(ignore_na=False, min_periods=0, adjust=True, span=12).mean()
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()

'''
　　Unit Root Test
   The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
   root, with the alternative that there is no unit root. That is to say the
   bigger the p-value the more reason we assert that there is a unit root
'''

def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Stataistic', 'P-value', '#Lags Used', 'Number of Obsevations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# 自相关和偏相关图，默认阶数为31阶
def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

# draw_ts(ts)
# draw_trend(ts, 12)
# draw_acf_pacf(ts)
# testStationarity(ts)


# 3. 平稳性处理
# a. 对数变换
ts_log = np.log(ts)
# draw_ts(ts_log)

# 以下采用的是12阶移动平均加上一阶差分
'''

# b. 平滑法
# draw_trend(ts_log, 12)
# c.  差分, 经过12阶差分和1阶差分
# 先用pandas将序列差分好，然后在对差分好的序列进行ARIMA拟合，只不过这样后面会多了一步人工还原的工作。
diff_12 = ts_log.diff(12)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
diff_12_1.dropna(inplace=True)
testStationarity(diff_12_1)

# d. 分解
from statsmodels.tsa.seasonal import seasonal_decompose
# statsmodels也支持两类分解模型，加法模型和乘法模型，这里我只实现加法，
# 乘法只需将model的参数设置为"multiplicative"即可。
decomposition = seasonal_decompose(ts_log, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
# plt.subplot(411)
# plt.plot(ts_log, label='Original')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal, label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residual')
# plt.legend(loc='best')
# plt.show()



# 4. 模型识别

# 该序列具有明显的年周期与长期成分。对于年周期成分我们使用
# 窗口为12的移动平进行处理，对于长期趋势成分我们采用1阶差分来进行处理。
rol_mean = ts_log.rolling(window=12).mean()
rol_mean.dropna(inplace=True)
ts_diff_1 = rol_mean.diff(1)
ts_diff_1.dropna(inplace=True)
testStationarity(ts_diff_1)

ts_diff_2 =  ts_diff_1.diff(1)
ts_diff_2.dropna(inplace=True)
testStationarity(ts_diff_2)
# draw_acf_pacf(ts_diff_2)

# 作者不使用ARIMA(ts_diff_1, order=(1, 1, 1))进行拟合，是因为含有差分操作时，预测结果还原老出问题，至今还没弄明白。
from statsmodels.tsa.arima_model import ARMA
model = ARMA(ts_diff_2, order=(1, 1))
result_arma = model.fit(disp=-1, method='css')


# 5. 样本拟合
predict_ts = result_arma.predict()
# 一阶差分还原
diff_shift_ts = ts_diff_1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
# 再次一阶差分还原
rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)
# 移动平均还原
rol_sum = ts_log.rolling(window=11).sum()
rol_recover = diff_recover*12 - rol_sum.shift(1)
# 对数还原
log_recover = np.exp(rol_recover)
log_recover.dropna(inplace=True)

# 使用均方根误差（RMSE）来评估模型样本内拟合的好坏。利用该准则进行判别时，需要剔除“非预测”数据的影响。
ts = ts[log_recover.index]      # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='red', label='Predict')
ts.plot(color='blue', label='Original')
plt.legend(loc='best')

plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - ts)**2) / ts.size))
plt.show()
'''

# 6. 完善 ARIMA 模型
# 前面提到statsmodels里面的ARIMA模块不支持高阶差分，我们的做法是将差分分离出来，
# 但是这样会多了一步人工还原的操作。基于上述问题，我将差分过程进行了封装，使
# 序列能按照指定的差分列表依次进行差分，并相应的构造了一个还原的方法，实现差分序列的自动还原。

# 差分操作
def diff_ts(ts, d):
    global shift_ts_list
    # 动态预测第二日的值时所需要的差分序列
    global last_data_shift_list
    shift_ts_list = []
    last_data_shift_list = []
    tmp_ts = ts
    for i in d:
        last_data_shift_list.append(tmp_ts[-i])
        # print(last_data_shift_list)
        shift_ts = tmp_ts.shift(i)
        shift_ts_list.append(shift_ts)
        tmp_ts = tmp_ts - shift_ts
    tmp_ts.dropna(inplace=True)
    return tmp_ts

# 还原操作
def predict_diff_recover(predict_value, d):
    if isinstance(predict_value, float):
        tmp_data = predict_value
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    elif isinstance(predict_value, np.ndarray):
        tmp_data = predict_value[0]
        for i in range(len(d)):
            tmp_data = tmp_data + last_data_shift_list[-i-1]
    else:
        tmp_data = predict_value
        for i in range(len(d)):
            try:
                tmp_data = tmp_data.add(shift_ts_list[-i-1])
            except:
                raise ValueError("What you input is not pd.Series type!")
        tmp_data.dropna(inplace=True)
    return tmp_data

'''
# 现在我们直接使用差分的方法进行数据处理，并以同样的过程进行数据预测与还原。
diffed_ts = diff_ts(ts_log, [12, 1])
model = arima_model(diffed_ts)      # arima_model是作者专门写的一个类
# ----------------------------------------------
# 法一：自己指定 p,q
# model.certain_model(1, 1)
# 法二：调用arima_model 类的 get_proper_model函数依据BIC准则识别模型的p,q值
model.get_proper_model()
# predict_ts = model.properModel.predict()
diff_recover_ts = predict_diff_recover(model.predict_ts, d=[12, 1])
log_recover = np.exp(diff_recover_ts)
ts = ts[log_recover.index]      # 过滤没有预测的记录
RMSE = np.sqrt(sum((log_recover - ts)**2) / ts.size)
print('BIC,p,q,RMSE', model.bic, model.p, model.q, RMSE)
# ----------------------------------------------


# ts = ts[log_recover.index]      # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='red', label='Predict')
ts.plot(color='blue', label='Original')
plt.legend(loc='best')

plt.title('RMSE: %.4f' % RMSE)
plt.show()
'''






# ---------------------------------------------------------------------------------------------------------------
def _add_new_data(ts, dat, type='day'):
    if type == 'day':
        new_index = ts.index[-1] + relativedelta(days=1)
    elif type == 'month':
        new_index = ts.index[-1] + relativedelta(months=1)
    ts[new_index] = dat


def add_today_data(model, ts,  data, d, type='day'):
    _add_new_data(ts, data, type)  # 为原始序列添加数据
    # 为滞后序列添加新值
    d_ts = diff_ts(ts, d)
    model.add_today_data(d_ts[-1], type)


def forecast_next_day_data(model, type='day'):
    if model == None:
        raise ValueError('No model fit before')
    fc = model.forecast_next_day_value(type)
    return predict_diff_recover(fc, [12, 1])


ts_train = ts_log[:'1956-12']
ts_test = ts_log['1957-1':]

diffed_ts = diff_ts(ts_train, [12, 1])
forecast_list = []

for i, dta in enumerate(ts_test):
    if i % 7 == 0:
        model = arima_model(diffed_ts)
        model.certain_model(1, 1)
    forecast_data = forecast_next_day_data(model, type='month')
    forecast_list.append(forecast_data)
    add_today_data(model, ts_train, dta, [12, 1], type='month')

predict_ts = pd.Series(data=forecast_list, index=ts['1957-1':].index)
log_recover = np.exp(predict_ts)
original_ts = ts['1957-1':]


# ts = ts[log_recover.index]      # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='red', label='Predict')
original_ts.plot(color='blue', label='Original')
plt.legend(loc='best')

plt.title('RMSE: %.4f' % np.sqrt(sum((log_recover - original_ts)**2) / ts.size))
plt.show()