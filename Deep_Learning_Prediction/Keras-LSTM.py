'''
!/usr/bin/env python

ref: https://daniel820710.medium.com/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Brief description:
LSTM implementation using Keras (v.0.3.2) with Theano (v.0.7.0) backend
This script is to train a model for stock market index prediction

*Input / Training data:
A time-series stock market close prices, e.g., prices in last 30 days
Data spliting ratio: train vs test = 9:1

*Output / Prediction:
predict the price in next day

*Usage:
Training:

python Keras-LSTM.py --past_days 7 --epoch 1500 --batch 768 > 7_E1500_B768_log.txt 2> 7_E1500_B768_errorlog.txt 

python Keras-LSTM.py --past_days 30 --epoch 1500 --batch 768 > 30_E1500_B768_log.txt 2> 30_E1500_B768_errorlog.txt 

python Keras-LSTM.py --past_days 90 --epoch 1500 --batch 256 > 90_E1500_B256_log.txt 2> 90_E1500_B256_errorlog.txt 

python Keras-LSTM.py --past_days 150 --epoch 1500 --batch 128 > 150_E1500_B128_log.txt 2> 150_E1500_B128_errorlog.txt 

Prediction:
python Keras-LSTM.py --past_days 30 --epoch 1500 --batch 768 -test ./Data/test_SSE_30past.csv --model_dir ./Outputs/30_E1500_B768/LSTM_model_30_E1500_B768.hdf5 -amend 7 --mode test > 30_E1500_B768_test7_log.txt 2> 30_E1500_B768_test7_errorlog.txt 
python Keras-LSTM.py --past_days 30 --epoch 1500 --batch 768 -test ./Data/test_SSE_30past.csv --model_dir ./Outputs/30_E1500_B768/LSTM_model_30_E1500_B768.hdf5 -amend 30 --mode test > 30_E1500_B768_test30_log.txt 2> 30_E1500_B768_test30_errorlog.txt 
python Keras-LSTM.py --past_days 30 --epoch 1500 --batch 768 -test ./Data/test_SSE_30past.csv --model_dir ./Outputs/30_E1500_B768/LSTM_model_30_E1500_B768.hdf5 -amend 90 --mode test > 30_E1500_B768_test30_log.txt 2> 30_E1500_B768_test30_errorlog.txt 

python Keras-LSTM.py --past_days 7 --epoch 1500 --batch 768 -test ./Data/test_SSE_7past.csv --model_dir ./Outputs/7_E1500_B768/LSTM_model_7_E1500_B768.hdf5 -amend 1 --mode test 
python Keras-LSTM.py --past_days 7 --epoch 1500 --batch 768 -test ./Data/test_SSE_7past.csv --model_dir ./Outputs/7_E1500_B768/LSTM_model_7_E1500_B768.hdf5 -amend 7 --mode test 
python Keras-LSTM.py --past_days 7 --epoch 1500 --batch 768 -test ./Data/test_SSE_7past.csv --model_dir ./Outputs/7_E1500_B768/LSTM_model_7_E1500_B768.hdf5 -amend 30 --mode test 
python Keras-LSTM.py --past_days 7 --epoch 1500 --batch 768 -test ./Data/test_SSE_7past.csv --model_dir ./Outputs/7_E1500_B768/LSTM_model_7_E1500_B768.hdf5 -amend 90 --mode test 


python Keras-LSTM.py --past_days 90 --epoch 1500 --batch 256 -test ./Data/test_SSE_90past.csv --model_dir ./Outputs/90_E1500_B256/LSTM_model_90_E1500_B256.hdf5 -amend 1 --mode test 
python Keras-LSTM.py --past_days 90 --epoch 1500 --batch 256 -test ./Data/test_SSE_90past.csv --model_dir ./Outputs/90_E1500_B256/LSTM_model_90_E1500_B256.hdf5 -amend 7 --mode test 
python Keras-LSTM.py --past_days 90 --epoch 1500 --batch 256 -test ./Data/test_SSE_90past.csv --model_dir ./Outputs/90_E1500_B256/LSTM_model_90_E1500_B256.hdf5 -amend 30 --mode test 
python Keras-LSTM.py --past_days 90 --epoch 1500 --batch 256 -test ./Data/test_SSE_90past.csv --model_dir ./Outputs/90_E1500_B256/LSTM_model_90_E1500_B256.hdf5 -amend 90 --mode test 


python Keras-LSTM.py --past_days 150 --epoch 1500 --batch 128 -test ./Data/test_SSE_150past.csv --model_dir ./Outputs/150_E1500_B128/LSTM_model_150_E1500_B128.hdf5 -amend 1 --mode test 
python Keras-LSTM.py --past_days 150 --epoch 1500 --batch 128 -test ./Data/test_SSE_150past.csv --model_dir ./Outputs/150_E1500_B128/LSTM_model_150_E1500_B128.hdf5 -amend 7 --mode test 
python Keras-LSTM.py --past_days 150 --epoch 1500 --batch 128 -test ./Data/test_SSE_150past.csv --model_dir ./Outputs/150_E1500_B128/LSTM_model_150_E1500_B128.hdf5 -amend 30 --mode test 
python Keras-LSTM.py --past_days 150 --epoch 1500 --batch 128 -test ./Data/test_SSE_150past.csv --model_dir ./Outputs/150_E1500_B128/LSTM_model_150_E1500_B128.hdf5 -amend 90 --mode test 

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
import os

# read csv file of data
# extract only close prices
def readData(path):
  train = pd.read_csv(path)
  # print train.head()
  # plt.plot(train['Close'][:200])
  # #plt.show()
  # plt.savefig('200_train_data_fig.png')
  # plt.clf()
  return train[['Close']].values.astype('float32')

# create dataset
# train with last pastDay-day close prices, predict close price in next day
# input: pastDay x 1 
# output: 1 x 1
def buildData(train, pastDay=20, futureDay=1):
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

# Split data to training set and test set(unseen data for model evaluation)
def splitData(X,Y,rate=0.1):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val


'''
Build a model of two-layer LSTM

layers = <feature-dim, window-size, 100, output-dim>
	feature-dim: 1 (only use close price)
	window-size: look_back days
	Layer 2 nodes: 100 (manual-defined)
	output-dim: 1 (only output close price)
	e.g., 1->7->100->1

Whole architecture:
in_feature_dim(x)  =>  window_size(w) -> dropout(0.4) => 100(n) -> dropout(0.3) => out_dim(y) -> activate(linear)

Loss: MSE
Optimizer: rmsprop

'''
def buildModel(layers):
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
	
'''
Prediction with a time-series input

Parameters:
	model: pretrained model
	dataset: ground truth data
	X_ordered: data to be predicted
	scaler: normalization range transformer, used to inverse transform predicted data back to original range
	look_back: window size, # past days for prediction
	predict_mode:
	set: prediction set, training set (train) or given test set (test)

'''
def predictNPlot(model, dataset, X_ordered, scaler, look_back, result_dir='./Results', amend_num = 1, set='train'):
	if amend_num > 1:
		past_days = X_ordered[0:1]
		predictions = model.predict(past_days)
		for idx in range(1, len(X_ordered)):
			if idx % amend_num == 0:
				past_days = X_ordered[idx: idx+1]
			else:
				past_days = np.append(past_days[0,1:,:], predictions[-1:, :]).reshape(1,look_back,1)
			predictions = np.concatenate((predictions, model.predict(past_days)))
	else:
		predictions = model.predict(X_ordered)
		
	print "predictions.shape = ", predictions.shape
	print "Saving "+ str(look_back) + '_' + set + '_' + str(amend_num) + 'interval'  + '_prediction.csv'
	predictions = scaler.inverse_transform(predictions)
	pd.DataFrame(predictions, columns=['Close']).to_csv(os.path.join(result_dir, str(look_back) + '_' + set + '_' + str(amend_num) + 'interval'  + '_prediction.csv'), index=None)

	# train predictions for plotting
	predictPlot = np.empty_like(dataset)
	predictPlot[:, :] = np.nan
	predictPlot[look_back:len(dataset), :] = predictions
	# plot baseline and predictions
	plt.title('Prediction on (%s) set in (%d)-day interval'%(set, amend_num))
	plt.plot(dataset, label='Ground Truth')
	plt.plot(predictPlot, label='Prediction')
	plt.legend()
	#plt.show()
	print "Saving "+ str(look_back) + '_' + set + '_' + str(amend_num) + 'interval'  + '_prediction_plot.png'
	plt.savefig(os.path.join(result_dir, str(look_back) + '_' + set + '_' + str(amend_num) + 'interval' + '_prediction_plot.png'))
	plt.clf()

	
if __name__ == '__main__':
	start_time = time.time()
	
	# Input parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_dir", "-i", default="./Data/train_SSE.csv", help="path to trainig data")
	parser.add_argument("--test_dir", "-test", default="./Data/test_SSE_30past.csv", help="path to testing data/unseen data")
	parser.add_argument("--output_dir", "-o", default="./Outputs", help="path to save model weights")
	parser.add_argument("--result_dir", default="./Results", help="path to save prediction results")
	parser.add_argument("--past_days", "-past", type=int, default=20, help="number of days of past days (window size): 7, 20, 30, 90, 150")
	parser.add_argument("--mode", default='train', help="train or test mode")
	parser.add_argument("--amend_num", "-amend", type=int, default=1, help="At inference, interval to correct real data, interval to make prediction")
	parser.add_argument("--model_dir", help="pretrained model directory, test mode only")
	parser.add_argument("--epoch", type=int, default=1500, help="Number of epochs")
	parser.add_argument("--batch", type=int, default=128, help="Batch size, e.g., 128 for 150 past days, 256 for 90 days, 768 for 30 past days")
	a = parser.parse_args()
	
	EPOCH = a.epoch # number of epochs
	BATCH_SIZE = a.batch #batch size	
	look_back = a.past_days #e.g, past day = 30, future day =1	
	output_dir = os.path.join(a.output_dir, '%d_E%d_B%d'%(look_back, EPOCH, BATCH_SIZE))
	result_dir = os.path.join(a.result_dir, '%d_E%d_B%d'%(look_back, EPOCH, BATCH_SIZE))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	
	'''
	Step 1:
	Read data
	Data processsing
	'''
	print '*****************************************************'
	dataset = readData(a.input_dir) # load dataset
	print "Dataset shape = ", dataset.shape
	
	# fix random seed for reproducibility
	np.random.seed(7)
	# normalize the dataset using all training data (fixed)
	scaler = MinMaxScaler(feature_range=(0, 1)) 
	train_norm = scaler.fit_transform(dataset)	
	
	
	if a.mode == 'train':
		# generate dataset with the last days(X) and future day(Y)
		X_ordered, Y_ordered = buildData(train_norm, look_back, 1) # many to one
		# shuffle training data
		X_train, Y_train = shuffle(X_ordered, Y_ordered)
		
		# split data to training and test set 
		# because no return sequence, Y_train and Y_val shape must be 2 dimension
		X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)
		print "X_train.shape, Y_train.shape, X_test.shape, Y_test.shape ="
		print X_train.shape, Y_train.shape, X_val.shape, Y_val.shape
		#e.g. (2036,30,1) (2036,1) (226,30,1)  (226,1)
		
	if a.test_dir is not None:
		testset = readData(a.test_dir)
		X_ordered_test, Y_ordered_test = buildData(scaler.transform(testset), look_back, 1)
		print "X_ordered_test.shape, Y_ordered_test.shape =", X_ordered_test.shape, Y_ordered_test.shape
	
	print '*****************************************************'

	'''
	Step 2
	Build LSTM model 
	Train and save model weights, OR load model weights
	'''
	#X_train.shape[2](feature-dim), window, 100(manual-defined), predict-dim
	model = buildModel([1, look_back, 100, 1]) #input_din, window/output_dim, output_dim, output_dim) 1=>20=>100=>1 
	
	
	if a.mode == 'train':
		print "Training preprocessing time elapsed: %.4f mins" %((time.time() - start_time)/60)
		history = model.fit(X_train, Y_train, nb_epoch=EPOCH, batch_size=BATCH_SIZE, validation_split=0.1)
		#history = model.fit(X_train, Y_train, nb_epoch=1500, batch_size=768, validation_data=(X_val, Y_val)) #or directly use test set for validation
		plt.plot(history.history['loss'], label='train')
		plt.plot(history.history['val_loss'], label='test')
		plt.title('Training Loss')
		plt.legend()
		plt.savefig(os.path.join(output_dir, '%d_training_loss_E%d_B%d.png'%(look_back, EPOCH, BATCH_SIZE)))
		plt.clf()
		#plt.show()
		
		print "Training time elapsed: %.4f mins" %((time.time() - start_time)/60)
			
		# save model weights
		model.save_weights(os.path.join(output_dir,"LSTM_model_%d_E%d_B%d.hdf5"%(look_back, EPOCH, BATCH_SIZE)), overwrite=True)
		print "Saved model to "+os.path.join(output_dir, "LSTM_model_%d_E%d_B%d.hdf5"%(look_back, EPOCH, BATCH_SIZE))
	else:
		# load model weights
		assert a.model_dir is not None
		model_dir = a.model_dir
		model.load_weights(model_dir)
		
	print '*****************************************************'

	'''
	Step 3
	Make prediction on training/test data
	Plot figure
	'''
	
	if a.mode == 'train':
		# make predictions
		print "Normalized data for evaluation:"
		trainScore = model.evaluate(X_train, Y_train, verbose=0)
		print 'Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore))
		testScore = model.evaluate(X_val, Y_val, verbose=0)
		print 'Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore))
		print '*************************'
		
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
		predictNPlot(model, dataset, X_ordered, scaler, look_back, result_dir=result_dir)
		
	if a.test_dir is not None:
		predictNPlot(model, testset, X_ordered_test, scaler, look_back, result_dir=result_dir, amend_num = a.amend_num, set='test')
	
	print '*****************************************************'
	print "Total time elapsed: %.4f mins" %((time.time() - start_time)/60)

	




