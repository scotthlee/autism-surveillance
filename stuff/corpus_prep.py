import numpy as np
import pandas as pd
import h5py

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import save_npz, load_npz

from tools import pad_integers

# Importing the data
indir = 'C:/data/addm/'
corpus = pd.read_csv(indir + 'corpus_with_lemmas_clean.csv')
text = corpus.dx
target = corpus.aucaseyn

# Vectorizing the text
vec = CountVectorizer(ngram_range=(1, 2), binary=False)
X = vec.fit_transform(text)

# Saving the docterm matrix as npz
save_npz(indir + 'doctermat.npz', X)

# Making a binary unigram doctermat for fastText
univec = CountVectorizer(ngram_range=(1, 1), binary=True)
X = univec.fit_transform(text)
save_npz(indir + 'unigram_doctermat.npz', X)

# Converting the docterm matrix to bag-of-integers for fastText
indices = [np.nonzero(row)[1] for row in X]
max_length = np.max([len(doc) for doc in indices])
int_sents = np.array([pad_integers(doc, max_length) for doc in indices])    

# Writing the integer sequences to HDF5
out = h5py.File(indir + 'unigram_ints.hdf5', mode='w')
out['sents'] = int_sents
out.close()

# Making a binary bigram document for fastText
bivec = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=5)
X = bivec.fit_transform(text)
save_npz(indir + 'bigram_doctermat.npz', X)

# Converting the docterm matrix to bag-of-integers for fastText
indices = [np.nonzero(row)[1] for row in X]
max_length = np.max([len(doc) for doc in indices])
int_sents = np.array([pad_integers(doc, max_length) for doc in indices])    

# Writing the integer sequences to HDF5
out = h5py.File(indir + 'bigram_ints.hdf5', mode='w')
out['sents'] = int_sents
out.close()

    
    
