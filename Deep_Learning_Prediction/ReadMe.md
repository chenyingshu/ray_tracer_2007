# Stock Market Prediction using LSTM in Vintage Implementation Environment

We use SSE index data from 1997 to 2006 as training data, and predict stock market date in the next two years (i.e., 2007 and 2008). 
We trained the LSTM (RNN) model using GPU accelaration via Theano.

## Specifications
OS: Windows 7 (64 bit)

GPU: 
  - Hardware: 1 x NVIDIA GeForece 8800 GTX 
  - CUDA driver version: 5.5
  - CUDA computing capability : 1.0 (CUDA <= v6)
  - 128 CUDA cores (8 cuda cores * 16 multiprocessors)
  - Max \#threads per multiprocessor = 768 

  VC Compiler: Microsoft Visual Studio 2010
  
Program language: Python 2.7
Python libraries:
  - Theano 0.7.0
  - Keras 0.3.2  

## Instructions 
I tested in Win 7 64bit only

### Quick setup

0. Set up Windows 7 OS 

1. Install MSVS 2010 and CUDA 5.5



2. Install Python environment

2.1. Install Anaconda 2.1.0

2.2. Install Dependencies

[optinal] Create your own python virtual environment:

create -n stock_pred
activate stock_pred

Install MinGW via conda:

Install Theano, Keras and other libraries:

Edit Theano config file in C:/Users/USER_NAME/.theanorc.txt

3. Clone source code
install git

git clone 
cd Deep_Learning_Prediction

4. Inference with pretrained model 
Two modes:

Non-iterative

Iterative
 
5. Training

## Results

## Algorithm and Implementation Details
Training data: 
X = <N, past_days, 1>
Y = <N, 1>

Network Architecture
ref: https://github.com/Kulbear/stock-prediction/blob/master/stock-prediction.ipynb

2-layer LSTM




