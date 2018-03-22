
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from IntensityGrid import IntensityGrid
np.random.seed(2018)

def create_training_seq(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

seq = np.loadtxt('earthquake_seq.txt')
marks = np.loadtxt('earthquake_mags.txt')

seq = np.reshape(seq,(len(seq),1))
train_seq = create_training_seq(seq,5)


scaler = MinMaxScaler(feature_range=(0,1))

training_seq = seq
training_seq = scaler.fit_transform(training_seq)

train_size = int(len(training_seq) * 0.67)
test_size = len(training_seq) - train_size
train, test = training_seq[0:train_size,:], training_seq[train_size:len(training_seq),:]

# reshape into X=t and Y=t+1
look_back = 10
trainX, trainY = create_training_seq(train, look_back)
testX, testY = create_training_seq(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(10, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=300, batch_size=1, verbose=1)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
QQpredseq = np.append(trainPredict,testPredict+trainPredict[-1])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(training_seq)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(training_seq)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(training_seq)-1, :] = testPredict

# plot baseline and predictions

plt.subplot(121)
plt.plot(scaler.inverse_transform(training_seq))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.suptitle('Sequence Prediction')
plt.ylabel('Interval Magnitude')
#plt.savefig('Sequence_Prediction.png')
#plt.close()

# Q-Q Plot

plt.subplot(122)
plt.plot(np.cumsum(seq),np.cumsum(seq), label='Original Sequence')
plt.plot(np.cumsum(seq[:len(QQpredseq)]),np.cumsum(QQpredseq),label='Predicted Sequence')
plt.suptitle('Q-Q Plot')
plt.savefig('Earthquake_Sequence_Prediction.png')
plt.close()

#plt.show()
