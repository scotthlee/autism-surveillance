import numpy as np
import pandas as pd

from scipy.sparse import load_npz

from stuff.tools import tfidf_to_counts, decompose

# Importing the data
filedir = 'C:/data/addm/'
seeds = np.array(pd.read_csv(filedir + 'seeds.csv')).flatten()
corpus = pd.read_csv(filedir + 'corpus_with_lemmas_clean.csv')
doctermat = load_npz(filedir + 'doctermat.npz')

# Setting the features and targets
X = np.array(doctermat.todense(), dtype=np.uint16)
y = np.array(corpus.aucaseyn, dtype=np.uint8)
n_range = range(corpus.shape[0])

# Doing the decommpositions
svd_10 = decompose(X, n_features=10)
svd_25 = decompose(X, n_features=25)
svd_50 = decompose(X, n_features=50)
svd_100 = decompose(X, n_features=100)
svd_200 = decompose(X, n_features=200)

# Saving the decompositions
np.save(filedir + 'svd_10', svd_10)
np.save(filedir + 'svd_25', svd_25)
np.save(filedir + 'svd_50', svd_50)
np.save(filedir + 'svd_100', svd_100)
np.save(filedir + 'svd_200', svd_200)
