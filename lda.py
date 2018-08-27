import numpy as np
import pandas as pd
import GPy, GPyOpt

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import load_npz

from tools import tfidf_to_counts, decompose
from metrics import binary_diagnostics

# Importing the data
filedir = 'C:/data/addm/'
seeds = np.array(pd.read_csv(filedir + 'seeds.csv')).flatten()
corpus = pd.read_csv(filedir + 'corpus_with_lemmas_clean.csv')

# Loading the decompositions
lda_5 = np.load(filedir + 'lda_5.npy')
lda_10 = np.load(filedir + 'lda_10.npy')
lda_15 = np.load(filedir + 'lda_15.npy')
lda_20 = np.load(filedir + 'lda_20.npy')
lda_30 = np.load(filedir + 'lda_30.npy')
lda_list = list([lda_5, lda_10, lda_15, lda_20, lda_30])

# Setting the features and targets
y = np.array(corpus.aucaseyn, dtype=np.uint8)
n_range = range(corpus.shape[0])

# Toggle for the optimization loop
optimize = True

if optimize:
    # Combinations of the two hyperparameters
    index_choices = np.array([0, 1, 2, 3, 4])
    c_choices = np.array([0.001, 0.01, 0.1, 1, 2, 8, 16])
    grid = np.meshgrid(index_choices, c_choices)
    indices = grid[0].flatten()
    c_params = grid[1].flatten()
    scores = np.zeros([len(c_params)])
    
    # Running the optimization
    train, test = train_test_split(n_range,
                                   test_size=0.3,
                                   stratify=y,
                                   random_state=10221983)
    
    for i in range(len(c_params)):
        X = lda_list[indices[i]]
        C = c_params[i]
        mod = LinearSVC(C=C)
        mod.fit(X[train], y[train])
        scores[i] = mod.score(X[test], y[test])
        print('Params were ' + str([indices[i], C]))
        print('Accuracy was ' + str(scores[i]) + '\n')
    
    best_i = np.argmax(scores)
    best = np.array([indices[best_i], c_params[best_i]])
    
    # Saving the best parameters to CSV
    pd.Series(best).to_csv(filedir + 'models/best_lda_params.csv',
                           index=False)

# Running the splits
best = pd.read_csv(filedir + 'models/best_lda_params.csv',
                   header=None)
best = np.array(best).flatten()
stats = pd.DataFrame(np.zeros([10, 15]))

# Decomposing to the optimal number of features
best_lda = lda_list[int(best[0])]
for i, seed in enumerate(seeds):
    train, test = train_test_split(n_range,
                                  stratify=y,
                                  random_state=seed,
                                  test_size=0.3)
    
    # Fitting the model
    mod = LinearSVC(C=best[1])
    print('Fitting model ' + str(i))
    mod.fit(best_lda[train], y[train])
    
    # Getting the predicted probs and thresholded guesses
    guesses = mod.predict(best_lda[test]).flatten()
    bin_stats = binary_diagnostics(y[test],
                                   guesses,
                                   accuracy=True)
    print(bin_stats)
    stats.iloc[i, :] = bin_stats.values

# Writing the output to CSV
stats.columns = ['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'ppv', 'npv',
                 'f1', 'acc', 'true', 'pred', 'abs', 'rel', 'mcnemar']
stats.to_csv(filedir + 'lda_stats.csv', index=False)
