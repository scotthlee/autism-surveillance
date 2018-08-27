import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.decomposition import LatentDirichletAllocation as LDA

from stuff.tools import tfidf_to_counts

# Importing the data
filedir = 'C:/data/addm/'
seeds = np.array(pd.read_csv(filedir + 'seeds.csv')).flatten()
corpus = pd.read_csv(filedir + 'corpus_with_lemmas_clean.csv')
doctermat = load_npz(filedir + 'unigram_doctermat.npz')

# Setting the features and targets
X = doctermat
y = np.array(corpus.aucaseyn, dtype=np.uint8)
n_range = range(corpus.shape[0])

# Doing the decommpositions
lda_5 = LDA(5, n_jobs=-1)
#x_5 = lda_5.fit_transform(X)
lda_10 = LDA(10, n_jobs=-1)
#x_10 = lda_10.fit_transform(X)
lda_15 = LDA(15, n_jobs=-1)
#x_15 = lda_15.fit_transform(X)
lda_20 = LDA(20, n_jobs=-1)
x_20 = lda_20.fit_transform(X)
lda_30 = LDA(30, n_jobs=-1)
x_30 = lda_30.fit_transform(X)

# Saving the decompositions
np.save(filedir + 'lda_5', x_5)
np.save(filedir + 'lda_10', x_10)
np.save(filedir + 'lda_15', x_15)
np.save(filedir + 'lda_20', x_20)
np.save(filedir + 'lda_30', x_30)

