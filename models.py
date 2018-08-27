'''
Custom models for document classification
'''
import pandas as pd
import numpy as np
import tensorflow as tf

from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, load_model
from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Concatenate, Multiply, Average, Add
from keras.layers import Input, Dense, Activation, Lambda, Softmax
from keras.layers import Embedding
from keras.layers import BatchNormalization, Dropout
from keras.layers import GlobalAveragePooling1D
from keras.engine.topology import Layer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from tools import *

# Basic version of fastText
def fastText(vocab_size,
             max_length,
             embedding_size=256,
             dropout=0.0):
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_size,
                        input_length=max_length))
    model.add(GlobalAveragePooling1D())
    if dropout != 0.0:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Basic version of slowText
def slowText(vocab_size,
             max_length,
             embedding_size=64,
             dropout=0.75):
    input = Input(shape=(max_length,))
    doc_rep = Dense(embedding_size)(input)
    if dropout != 0.0:
        doc_rep = Dropout(dropout)(doc_rep)
    output = Dense(1, activation='sigmoid')(doc_rep)
    model = Model(input, output)
    return model    

# Main class for the NB-SVM
class NBSVM:
    def __init__(self, C=0.1, beta=0.95):
        self.beta = beta
        self.C = C
    
    # Fits the model to the data and does the interpolation
    def fit(self, x, y):
        # Calculating the log-count ratio
        X_pos = x[np.where(y == 1)]
        X_neg = x[np.where(y == 0)]
        self.r = log_count_ratio(X_pos, X_neg)
        X = np.multiply(self.r, x)
        
        # Training linear SVM with NB features but no interpolation
        svm = LinearSVC(C=self.C)
        svm.fit(X, y)
        self.coef_ = svm.coef_
        self.int_coef_ = interpolate(self.coef_, self.beta)
        self.bias = svm.intercept_
    
    # Scores the interpolated model
    def score(self, x, y):
        X = np.multiply(self.r, x)
        return accuracy(X, y, self.int_coef_, self.bias)
    
    # Returns binary class predictions
    def predict(self, x):
        X = np.multiply(self.r, x)
        return linear_prediction(X, self.int_coef_, self.bias)

# Same as above, only without interpolation (i.e. just a linear SVM
# with naive Bayes features)
class simpleNBSVM:
    def __init__(self, C=0.1):
        self.C = C
    
    # Fits the model to the data and does the interpolation
    def fit(self, x, y):
        # Calculating the log-count ratio
        X_pos = x[np.where(y == 1)]
        X_neg = x[np.where(y == 0)]
        self.r = log_count_ratio(X_pos, X_neg)
        X = np.multiply(self.r, x)
        
        # Training linear SVM with NB features but no interpolation
        svm = LinearSVC(C=self.C)
        self.mod = CalibratedClassifierCV(svm)
        self.mod.fit(X, y)
    
    # Scores the interpolated model
    def score(self, x, y):
        X = np.multiply(self.r, x)
        return self.mod.score(X, y)
    
    # Returns binary class predictions
    def predict(self, x):
        X = np.multiply(self.r, x)
        return self.mod.predict(X)
    
    # Returns predicted class probabilities
    def predict_proba(self, x):
        X = np.multiply(self.r, x)
        return self.mod.predict_proba(X)
