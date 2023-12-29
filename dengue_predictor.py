# -*- coding: utf-8 -*-


# !sudo apt-get install python3.7
# !sudo apt-get update -y
# !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
# !sudo update-alternatives --config python3
# !sudo apt install python3-pip
# !python -m pip install --upgrade --force-reinstall pip

# !sudo apt-get install python3.7-distutils
# !sudo apt-get install python3-apt
# !python --version

# Connect google drive
from google.colab import drive
drive.mount('/content/drive')

!pip install chart_studio

import pandas as pd
import seaborn as sns
from matplotlib import figure
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

df = pd.read_csv("/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv")

ddf = df.drop(['week'],axis=1)
ddf.corr()
sns.heatmap(ddf.corr(),cmap="crest",xticklabels="auto", yticklabels="auto",linewidth=".05",annot=True,vmin=-1,vmax=1)

from operator import index
df.plot(xlabel="week",xticks=df['week'], linestyle='--',linewidth=2,fillstyle='full',)

# pp = sns.pairplot(data=ddf, aspect=1,hue='pahang',height=4)
# pp.savefig('pp.pdf')

"""
# LSTM Related Resources

---


"""

# !pip install wandb


import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
# import wandb
# from wandb.keras import WandbCallback
import seaborn as sns
sns.set(rc = {'figure.figsize':(15,8)})
warnings.filterwarnings('ignore')

denggie = pd.read_csv('/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv', parse_dates=['end_date'])
denggie.head()



df = denggie[['end_date','pahang']].copy()
df.head()
df.info()

df = df.iloc[:,1].values
plt.plot(df)
plt.xlabel("Number of Week")
plt.ylabel("Number of case")
plt.title("Weekly Case in Pahang")
plt.show()


df = df.reshape(-1,1)
df = df.astype("float32")
df.shape

scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(df)

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train = df[0:train_size,:]
test = df[train_size:len(df),:]
print("train size: {}, test size: {} ".format(len(train), len(test)))

import numpy as np

time_stamp = 9

dataX = []
dataY = []

for i in range(len(train)-time_stamp-1):
    a = train[i:(i+time_stamp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stamp, 0])

trainX = np.array(dataX)
trainY = np.array(dataY)

dataX = []
dataY = []
for i in range(len(test)-time_stamp-1):
    a = test[i:(i+time_stamp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stamp, 0])
testX = np.array(dataX)
testY = np.array(dataY)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf




es = EarlyStopping(monitor='loss', mode='auto', patience=3, verbose=1)
model = Sequential()
model.add(Bidirectional(LSTM(10, input_shape=(1, time_stamp),return_sequences=False))) # 10 lstm neuron
model.add(Dense(30))
model.add(Dropout(.001,noise_shape=None,seed=None))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','accuracy'])

history = model.fit(trainX, trainY,validation_split=0.33, epochs=5000, batch_size=1, callbacks=[es])


model.summary()

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
testScore_mse = mean_squared_error(testY[0], testPredict[:,0])
print('Test Score: %.2f MSE' % (testScore_mse))
testScore_mae = mean_absolute_error(trainY[0], trainPredict[:,0])
print('Test Score: %.2f MAE' % (testScore_mae))
testScore_mape = mean_absolute_percentage_error(trainY[0], trainPredict[:,0])
print('Test Score: %.2f MAPE' % (testScore_mape))

# shifting train
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stamp:len(trainPredict)+time_stamp, :] = trainPredict

# shifting test
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(time_stamp*2)+2:len(df), :] = testPredict

# import seaborn as sns
# sns.set(rc = {'figure.figsize':(15,8)})
plt.plot(scaler.inverse_transform(df),label="Actual Data")
plt.plot(trainPredictPlot,label="Train predict",linestyle='--',linewidth=2)
plt.plot(testPredictPlot,label="Test Predict",linestyle='--',linewidth=2)
plt.xlabel("Number of Week")
plt.ylabel("Number of case")
plt.title("LSTM- Actual vs predicted case")
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("Number of Epochs")
plt.ylabel("Loss value")
plt.legend(['Train','Test'])
plt.show()

"""#SARIMA Related Resources"""



!python --version
!pip install pmdarima





from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from numpy import log

df = pd.read_csv("/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv")
df.head()

ddf = df[['end_date','pahang']].copy()
ddf.info()
ddf['end_date'] = pd.to_datetime(ddf['end_date'])
ddf.info()
result = adfuller(df.pahang.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

ddf.set_index('end_date', inplace=True)
# ddf.index = pd.DatetimeIndex(ddf.index).to_period('M')
ddf.index.asfreq='W'
ddf['pahang'].plot(figsize=(12,5));

auto_arima(ddf['pahang'],seasonal=True).summary()

ddf.shape

train = ddf.iloc[:124]
test = ddf.iloc[124:]

import numpy as np
model = SARIMAX(train['pahang'],order=(1,1,1))
results = model.fit()
results.summary()
print(f"SARIMA MAE: {results.mae}")
print(f"SARIMA MSE: {results.mse}")
print(f"SARIMA RMSE: {np.sqrt(results.mse)}")

start=len(train)
end=len(train)+len(test)-1
predictions = results.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMAX(1,1,1) Predictions')

model = SARIMAX(ddf['pahang'],order=(1,1,1))
results = model.fit()
fcast = results.predict(len(ddf),len(ddf)+4,typ='levels').rename('SARIMA(1,1,1) Forecast')
print(f"SARIMA MAE: {results.mae}")
print(f"SARIMA MSE: {results.mse}")
print(f"SARIMA RMSE: {np.sqrt(results.mse)}")
MAPE = np.mean(np.abs((ddf['pahang'] - predictions) / ddf['pahang'])) * 100
print(f"SARIMA MAPE: {MAPE}")

title = 'Weekly denggie Prediction'
ylabel='Pahang case'
xlabel='Date'

ax = ddf['pahang'].plot(legend=True,figsize=(12,6),title=title)
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

"""#ARIMA"""

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams.update({'figure.figsize':(15,8), 'figure.dpi':120})

# Import data
odf = pd.read_csv('/content/drive/MyDrive/dengue_dashboard/dengue_m_ds_2021.csv')
df = odf[['end_date','pahang']].copy()

# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.pahang); axes[0, 0].set_title('Original Series')
plot_acf(df.pahang, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(df.pahang.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.pahang.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(df.pahang.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.pahang.diff().diff().dropna(), ax=axes[2, 1])

plt.show()

from pmdarima.arima.utils import ndiffs
y = df.pahang

## Adf Test
ndiffs(y, test='adf')  # 2

# KPSS test
ndiffs(y, test='kpss')  # 0

# PP test:
ndiffs(y, test='pp')  # 2

# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.pahang.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.pahang.diff().dropna(), ax=axes[1])

plt.show()

from statsmodels.tsa.arima.model import ARIMA


model = ARIMA(df.pahang, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
from statsmodels.graphics.tsaplots import plot_predict
fcast = model_fit.predict(len(df),len(df)+4,typ='levels').rename('ARIMA(1,1,1) Forecast')
fcast.plot(legend=True)


from statsmodels.tsa.stattools import acf

# Create Training and Test
train = df.pahang[:42]
test = df.pahang[42:]


model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit()

# Forecast
test_test = fitted.forecast(15, alpha=0.05)  # 95% conf

# print(test_test)
# Make as pandas series
fc_series = pd.Series(test_test.iloc[0], index=test.index)
lower_series = pd.Series(test_test.iloc[1], index=test.index)
upper_series = pd.Series(test_test.iloc[1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series,
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    # corr = np.corrcoef(forecast, actual)[0,1]   # corr
    # mins = np.amin(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    # maxs = np.amax(np.hstack([forecast[:,None], actual[:,None]]), axis=1)
    # minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(test_test.iloc[0]-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1
            })

forecast_accuracy(test_test.iloc[0], test.values)

import pmdarima as pm

model = pm.auto_arima(df.pahang, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(model.summary())


model.plot_diagnostics(figsize=(7,7))
plt.show()

# Forecast
n_periods = 4
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.pahang), len(df.pahang)+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df.pahang)
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index,
                 lower_series,
                 upper_series,
                 color='k', alpha=.15)

plt.title("Forecast for next 4 weeks")
plt.show()

"""#SVR related resources"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import seaborn as sns

data = pd.read_csv('/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv')
data.head()

df = data[['end_date','pahang']].copy()
df['end_date'] = pd.to_datetime(df['end_date'])

df.info()
df.head()

df.describe()

df['week number'] = [i for i in range(1,len(df['end_date'])+1)]
df.tail()

# Creating Train And Target Data
# train Data
week = df['week number'].values.reshape(len(df['week number']),1)
# Target Data
pahang = df['pahang']

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
import numpy as np

svr = SVR(kernel="rbf",C = 1e3,gamma = 0.1)
svr.fit(week,pahang)
pred = svr.predict(week)

f_week = [[135],[136],[137],[138],[139]]
future = svr.predict(f_week)

print("R^2 : ", r2_score(pahang, pred))
print("MAE :", mean_absolute_error(pahang,pred))
print("MSE:", mean_squared_error(pahang,pred))
print("RMSE:", np.sqrt(mean_squared_error(pahang,pred)))
print("MAPE:", mean_absolute_percentage_error(pahang,pred))


plt.figure(figsize=(15,5))
plt.scatter(week,pahang,label ='Original Case',color="red")
plt.plot(week,pred,'b',marker='*',label ='Predicted Case')
plt.plot(f_week,svr.predict(f_week),'g',marker='o',label ='Futre Predicted Case')
plt.xlabel("Number of week")
plt.ylabel("Number of case")
plt.title("SVR- Actual vs predicted case")
plt.legend()

plt.figure(figsize=(15,5))

X= week
y= pahang
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='red', label='original Case')
# plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('Number of week')
plt.ylabel('Number of case')
plt.title('Different kernels of SVR')
plt.legend()
plt.show()

"""#RFR Related Resources"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error
import numpy as np



data = pd.read_csv('/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv')
data.head()
df = data[['end_date','pahang']].copy()
df['end_date'] = pd.to_datetime(df['end_date'])

df.info()
df.head()
df.describe()
df['week number'] = [i for i in range(1,len(df['end_date'])+1)]
df.tail()

# Creating Train And Target Data
# train Data
week = df['week number'].values.reshape(len(df['week number']),1)
# Target Data
pahang = df['pahang']



rfr = RandomForestRegressor(max_depth=4, random_state=8)
rfr.fit(week,pahang)
pred = rfr.predict(week)

f_week = [[135],[136],[137],[138],[139]]
future = rfr.predict(f_week)

print("R^2 : ", r2_score(pahang, pred))
print("MAE :", mean_absolute_error(pahang,pred))
print("MSE:", mean_squared_error(pahang,pred))
print("RMSE:", np.sqrt(mean_squared_error(pahang,pred)))
print("MAPE:", mean_absolute_percentage_error(pahang,pred))

# plotting prediction and real values
plt.figure(figsize=(15,5))
plt.scatter(week,pahang,label ='Original Case', color="red")
plt.plot(week,pred,'b',marker='*',label ='Predicted Case')
plt.plot(f_week,rfr.predict(f_week),'g',marker='x',label ='Futre Predicted Case')
plt.xlabel("Number of week")
plt.ylabel("Number of case")
plt.title("RFR- Actual vs predicted case")
plt.axes()
plt.legend()
print(future)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor

sns.set()

X, y = week,pahang

# Train classifiers
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)

ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
ereg.fit(X, y)

xt = X[:20]

pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = ereg.predict(xt)

plt.figure(figsize=(15,5))
plt.plot(pred1, "gd", label="GradientBoostingRegressor")
plt.plot(pred2, "b^", label="RandomForestRegressor")
plt.plot(pred3, "ys", label="LinearRegression")
plt.plot(pred4, "r*", ms=10, label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()

score1 = reg1.score(X,y)
print("R-squared (GB):", score1)
mae1= mean_absolute_error(xt, pred1)
mse1 = mean_squared_error(xt, pred1)
mape1 = mean_absolute_percentage_error(xt, pred1)
print("MAE (GB): ", mae1)
print("MSE (GB): ", mse1)
print("RMSE (GB): ", mse1*(1/2.0))
print("MAPE (GB): ", mape1)

score2 = reg2.score(X,y)
print("R-squared (RFR):", score2)
mae2 = mean_absolute_error(xt, pred2)
mse2 = mean_squared_error(xt, pred2)
mape2 = mean_absolute_percentage_error(xt, pred2)
print("MAE (RFR): ", mae2)
print("MSE (RFR): ", mse2)
print("RMSE (RFR): ", mse2*(1/2.0))
print("MAPE (RFR): ", mape2)


score3 = reg3.score(X,y)
print("R-squared (LR):", score3)
mae3 = mean_absolute_error(xt, pred3)
mse3 = mean_squared_error(xt, pred3)
mape3 = mean_absolute_percentage_error(xt, pred3)
print("MAE (LR): ", mae3)
print("MSE (LR): ", mse3)
print("RMSE (LR): ", mse3*(1/2.0))
print("MAPE (LR): ", mape3)

score4 = ereg.score(X,y)
print("R-squared (VR):", score4)
mae4 = mean_absolute_error(xt, pred4)
mse4 = mean_squared_error(xt, pred4)
mape4 = mean_absolute_percentage_error(xt, pred4)
print("MAE (VR): ", mae4)
print("MSE (VR): ", mse4)
print("RMSE (VR): ", mse4*(1/2.0))
print("MAPE (VR): ", mape4)

"""#Prophet Related resources"""

!pip install pystan==2.19.1.1 prophet

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from prophet import Prophet
import plotly.express as px
import plotly.offline as py
py.init_notebook_mode()
# %matplotlib inline

df = pd.read_csv("/content/drive/MyDrive/dengue_dashboard/dataset-3y-cleaned-no-missing.csv")
df.head()
ddf= df[['end_date','pahang']].copy()
ddf.head()

ddf.info()

ddf['end_date'] = pd.to_datetime(ddf['end_date'])
ddf.info()
ddf.head()

# ddf.set_index('end_date', inplace=True)
# ddf.index = pd.DatetimeIndex(ddf.index).to_period('M')
# ddf.index.asfreq='W'

fig = px.line(ddf, x="end_date", y="pahang", title='Denggi Case Time Series for Phanag',markers=None)
fig.update_xaxes(rangeslider_visible=True)
fig.show(renderer="colab")

plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
pd.plotting.register_matplotlib_converters()
pdf.set_index('ds').y.plot(marker='X')
pdf['y'] = np.log(pdf['y'])
pdf.head()
pdf.set_index('ds').y.plot(marker="X").get_figure()

pdf = ddf.rename(columns={'end_date':'ds', 'pahang':'y'})
model = Prophet(weekly_seasonality=False,daily_seasonality=True,yearly_seasonality=False,interval_width=0.95)
model.fit(pdf)

future = model.make_future_dataframe(periods=6,freq='W')
forecast = model.predict(future)

forecast.tail()

from prophet.plot import add_changepoints_to_plot
fig1 = model.plot(forecast)
# fig1 = model.plot(legend)
a= add_changepoints_to_plot(fig1.gca(),model,forecast)

fig2 = model.plot_components(forecast)

fig = px.line(pdf, x='ds', y='y')
fig.update_xaxes(rangeslider_visible=True)
fig.show(renderer="colab")
# pdf.head()
# pdf.info()

from sklearn.model_selection import train_test_split

train_data = pdf.sample(frac=0.8, random_state=10)
validation_data = pdf.drop(train_data.index)

print(f'training data size : {train_data.shape}')
print(f'validation data size : {validation_data.shape}')

train_data = train_data.reset_index()
validation_data = validation_data.reset_index()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
model = Prophet(weekly_seasonality=True,daily_seasonality=False,yearly_seasonality=False,interval_width=0.95)
model.fit(train_data)

prediction = model.predict(pd.DataFrame({'ds':validation_data['ds']}))
y_actual = validation_data['y']
y_predicted = prediction['yhat']
y_predicted = y_predicted.astype(int)
mae = mean_absolute_error(y_actual, y_predicted)
mse = mean_squared_error(y_actual, y_predicted)
r2 = r2_score(y_actual, y_predicted)
print(f"Prophet MAE: {mae} \n Prophet MSE: {mse} \n Prophet R2: {r2}")

from plotly.subplots import make_subplots
import plotly.graph_objs as go
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=validation_data['ds'], y=y_actual, name="actual targets"),
    secondary_y=False,)
fig.add_trace(
    go.Scatter(x=validation_data['ds'], y=y_predicted, name="predicted targets"),
    secondary_y=True,)
fig.update_layout(
    title_text="Actual vs Predicted Targets")
fig.update_xaxes(title_text="Timeline")
fig.update_yaxes(title_text="actual targets", secondary_y=False)
fig.update_yaxes(title_text="predicted targets", secondary_y=True)
fig.show(renderer='colab')

d = {'col1': ['2022-01-09','2022-01-16','2022-01-23','2022-01-30']}
df = pd.DataFrame(data=d)
test_prediction = model.predict(pd.DataFrame({'ds':df['col1']}))
# print(test_prediction['yhat'])
for item in test_prediction['yhat']:
  print(int(item))

from prophet.diagnostics import cross_validation
df_cv = cross_validation(model, initial='300 days', period='1 day', horizon = '28 days')
df_cv.head()

from prophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()

