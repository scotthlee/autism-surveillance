'''
Support functions for the document classification models.
'''
import numpy as np
import pandas as pd

from sys import getsizeof
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from keras import backend as K
from copy import deepcopy

# Releases GPU memory for retraining Keras models
def limit_mem():
    K.get_session().close()
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))

# Calculates the log-count ratio r
def log_count_ratio(pos_text, neg_text, alpha=1):
    p = np.add(alpha, np.sum(pos_text, axis=0))
    q = np.add(alpha, np.sum(neg_text, axis=0))
    p_norm, q_norm = np.sum(p), np.sum(q)
    p_ratio = np.true_divide(p, p_norm)
    q_ratio = np.true_divide(q, q_norm)
    r = np.log(np.true_divide(p_ratio, q_ratio))
    return r

# Returns interpolated weights for constructing the NB-SVM
def interpolate(w, beta):
	return ((1 - beta) * (np.sum(w) / w.shape[1])) + (beta * w)

# Returns the predicted outputs based on inputs, training weights, and bias
# exp=True will exponentiate the predicted values, transforming to [0, 1]
def linear_prediction(x, w, b, neg=0, binary=True):
    guesses = np.matmul(x, w.transpose()) + b
    if binary:
        prediction = np.array(np.sign(guesses), dtype=np.int8)
        if neg == 0:
            prediction[prediction == -1] = 0
    else:
        prediction = guesses
    return prediction

# Gets the size of an object in memory in GB
def gigs(obj):
    return np.true_divide(getsizeof(obj), 1e9)
    
#converts tf-idf matrices to binary count matrices
def tfidf_to_counts(data):
	data = deepcopy(data)
	data[np.where(data > 0)] = 1
	return data

# Returns the SVD of a document-term matrix
def decompose(doc_vecs, n_features=100, normalize=False, flip=False):
	svd = TruncatedSVD(n_features)	
	if normalize:	
		if flip:
			lsa = make_pipeline(svd, Normalizer(copy=False))
			doc_mat = lsa.fit_transform(doc_vecs.transpose())
			doc_mat = doc_mat.transpose()
		else:
			lsa = make_pipeline(svd, Normalizer(copy=False))		
			doc_mat = lsa.fit_transform(doc_vecs)
	else:
		if flip:
			doc_mat = svd.fit_transform(doc_vecs.transpose())
			doc_mat = doc_mat.transpose()
		else:
			doc_mat = svd.fit_transform(doc_vecs)
	return doc_mat
		
# Converts a list of tokens to an array of integers
def to_integer(tokens, vocab_dict, encode=False,
               subtract_1=False, dtype=np.uint32):
    if encode:
        tokens = [str(word, errors='ignore') for word in tokens]
    out = np.array([vocab_dict.get(token) for token in tokens], dtype=dtype)
    if subtract_1:
        out = out - 1
    return out

# Pads a 1D sequence of integers (representing words)
def pad_integers(phrase, max_length, padding=0):
    pad_size = max_length - len(phrase)
    padded = np.concatenate((phrase, np.repeat(padding, pad_size)))
    return padded
