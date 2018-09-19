
from pandas import read_csv
from pandas import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

def parser(x):
	return datetime.strptime(x + "", '%d/%m/%Y')
 
series = read_csv('test.csv', parse_dates=[0, 1], index_col=[0,1], usecols=[0,1,3])
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = SARIMAX(history, order=(5,1,0) seasonal_order=(1,0,0,3))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
