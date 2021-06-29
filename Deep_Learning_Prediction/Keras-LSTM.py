'''
!/usr/bin/env python

ref: https://daniel820710.medium.com/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Brief description:
LSTM implementation using Keras (v.0.3.2) with Theano (v.0.7.7) backend
This script is to train a model for stock market index prediction

*Input / Training data:
A time-series stock market close prices, e.g., prices in last 30 days
Data spliting ratio: train vs validation = 9:1

*Output / Prediction:
predict the price in next day

*Usage:
python Keras-LSTM.py -i train_SSE.csv

By Yingshu Chen in June 2021
'''

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import time
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
# get_ipython().magic(u'matplotlib inline')

# read csv file of training data
# extract only close prices
def readTrain(path):
  train = pd.read_csv(path)
  # print train.head()
  # plt.plot(train['Close'][:200])
  # #plt.show()
  # plt.savefig('200_train_data_fig.png')
  # plt.clf()
  return train[['Close']].values.astype('float32')

# create dataset
# train with last 30-day close prices, predict close price in 31st day
# input: 30x1 
# output: 1x1
def buildTrain(train, pastDay=20, futureDay=1):
  X_train, Y_train = [], []
  for i in range(len(train)-futureDay-pastDay+1):
    X_train.append(train[i:i+pastDay])
    Y_train.append(train[i+pastDay:i+pastDay+futureDay][0])
  return np.array(X_train), np.array(Y_train)


# shuffling
def shuffle(X,Y):
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

def splitData(X,Y,rate=0.1):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val



def buildManyToOneModel(shape):
  model = Sequential()
  model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

  # two-layer LSTM
def build_model(layers):
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : %.4f mins" %((time.time() - start)/60))
    return model
	
def buildManyToManyModel(shape):
  model = Sequential()
  model.add(LSTM(9, input_length=shape[1], input_dim=shape[2], return_sequences=True))
  # output shape: (5, 1)
  model.add(TimeDistributed(Dense(1)))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model



def predictNPlot(model, dataset, X_ordered, scaler, look_back, predict_mode = 'non-iter', set='train'):
	if predict_mode != 'non-iter':
		past_days = X_ordered[0:1]
		predictions = model.predict(past_days)
		for idx in range(1, len(X_ordered)):
			past_days = np.append(past_days[0,1:,:], predictions[-1:, :]).reshape(1,look_back,1)
			predictions = np.concatenate((predictions, model.predict(past_days)))
	else:
		predictions = model.predict(X_ordered)
		
	print "predictions.shape = ", predictions.shape
	predictions = scaler.inverse_transform(predictions)
	pd.DataFrame({'Close': predictions}).to_csv(str(look_back) + '_' + set + '_' + predict_mode + '_prediction.csv', index=None)

	# train predictions for plotting
	predictPlot = np.empty_like(dataset)
	predictPlot[:, :] = np.nan
	predictPlot[look_back:len(dataset), :] = predictions
	# plot baseline and predictions
	plt.title('Prediction on (%s) set in (%s) mode'%(set, predict_mode))
	plt.plot(dataset, label='Ground Truth')
	plt.plot(predictPlot, label='Prediction')
	plt.legend()
	#plt.show()
	plt.savefig(str(look_back) + '_' + set + '_' + predict_mode + '_prediction_plot.png')
	plt.clf()

if __name__ == '__main__':
	start_time = time.time()
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", "-i", default="train_SSE.csv", help="path to trainig data")
	parser.add_argument("--test_dir", "-test", default="test_SSE_20past.csv", help="path to testing data/unseen data")
	parser.add_argument("--output_dir", "-o", default="Many2One_Model", help="path to save model weights")
	parser.add_argument("--past_days", "-past", type=int, default=20, help="number of days of past days")
	parser.add_argument("--mode", default='train', help="train or test mode")
	parser.add_argument("--test_mode", default='non-iter', help="non-iter or iter (iterative) prediction")
	parser.add_argument("--model_dir", help="pretrained model directory, test mode only")
	a = parser.parse_args()
	
	'''
	Step 1:
	Read data
	Data processsing
	'''
	look_back = a.past_days #e.g, past day = 30, future day =1	
	dataset = readTrain(a.input_dir) # load dataset
	print "dataset shape = ", dataset.shape
	
	# fix random seed for reproducibility
	np.random.seed(7)
	# normalize the dataset
	scaler = MinMaxScaler(feature_range=(0, 1)) 
	train_norm = scaler.fit_transform(dataset)	
	
	
	if a.mode == 'train':
		# generate dataset with the last days(X) and future day(Y)
		X_ordered, Y_ordered = buildTrain(train_norm, look_back, 1) # many to one
		# shuffle training data
		X_train, Y_train = shuffle(X_ordered, Y_ordered)
		
		# split data to training and validataiton set 
		#because no return sequence, Y_train and Y_val shape must be 2 dimension
		X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
		print "X_train.shape, Y_train.shape, X_val.shape, Y_val.shape = "
		print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape
		#(2036,30,1) (2036,1) (226,30,1)  (226,1)
		
	if a.test_dir is not None:
		testset = readTrain(a.test_dir)
		X_ordered_test, Y_ordered_test = buildTrain(scaler.transform(testset), look_back, 1)
		print "X_ordered_test.shape, Y_ordered_test.shape = ", X_ordered_test.shape, Y_ordered_test.shape
	
	'''
	Step 2
	Build LSTM model 
	Train and save model weights, OR load model weights
	'''
	#X_train.shape[2](feature-dim), window, 100(manual-defined), predict-dim
	model = build_model([1, look_back, 100, 1]) #input_din, window/output_dim, output_dim, output_dim) 1=>20=>100=>1 
	
	
	if a.mode == 'train':
		#callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")	
		print "Training preprocessing time elapsed: %.4f mins" %((time.time() - start_time)/60)
		#history = model.fit(X_train, Y_train, nb_epoch=1500, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
		history = model.fit(X_train, Y_train, nb_epoch=1500, batch_size=128, validation_split=0.1)
		#history = model.fit(X_train, Y_train, nb_epoch=1500, batch_size=768, validation_data=(X_val, Y_val))
		plt.plot(history.history['loss'], label='train')
		plt.plot(history.history['val_loss'], label='test')
		plt.title('')
		plt.legend()
		plt.savefig(str(look_back) + '_training_loss.png')
		plt.clf()
		#plt.show()
		
		print "Training time elapsed: %.4f mins" %((time.time() - start_time)/60)
			
		# save model weights
		model.save_weights(a.output_dir+"_"+str(look_back), overwrite=True)
	else:
		# load model weights
		if a.model_dir is not None:
			model_dir = a.model_dir
		else: model_dir = a.output_dir+"_"+str(look_back)
		model.load_weights(model_dir)
	
	'''
	Step 3
	Make prediction on training/validation data
	Plot figure
	'''
	
	if a.mode == 'train':
		# make predictions
		print "Normalized data for evaluation:"
		trainScore = model.evaluate(X_train, Y_train, verbose=0)
		print 'Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore))
		testScore = model.evaluate(X_val, Y_val, verbose=0)
		print 'Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore))
		print '*****************************************************'
		
		print "Original ranged data for evaluation: "
		trainPredict = model.predict(X_train)
		testPredict = model.predict(X_val)
		print "trainPredict.shape, testPredict.shape = ", trainPredict.shape, testPredict.shape
		# invert predictions
		trainPredict = scaler.inverse_transform(trainPredict)
		trainY = scaler.inverse_transform(Y_train)
		testPredict = scaler.inverse_transform(testPredict)
		testY = scaler.inverse_transform(Y_val)
		print "trainY.shape, trainPredict.shape, testY.shape, testPredict.shape = "
		print trainY.shape, trainPredict.shape, testY.shape, testPredict.shape
		# calculate root mean squared error
		trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:,0]))
		print('Train Score: %.2f RMSE' % (trainScore))
		testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
		print('Test Score: %.2f RMSE' % (testScore))
		

	if a.mode == 'train':
		predictNPlot(model, dataset, X_ordered, scaler, look_back)
		
	if a.test_dir is not None:
		predictNPlot(model, testset, X_ordered_test, scaler, look_back, predict_mode = a.test_mode, set='test')
	
	print "Total time elapsed: %.4f mins" %((time.time() - start_time)/60)

	




