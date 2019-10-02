import numpy as np
import pandas as pd
import GPy, GPyOpt

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss as brier_score
from sklearn.metrics import accuracy_score, f1_score
from scipy.sparse import load_npz

from stuff.models import NBSVM, simpleNBSVM
from stuff.tools import tfidf_to_counts
from stuff.metrics import binary_diagnostics

# Importing the data
filedir = 'C:/data/addm/'
seeds = np.array(pd.read_csv(filedir + 'seeds.csv')).flatten()
corpus = pd.read_csv(filedir + 'corpus_with_lemmas_clean.csv')
doctermat = load_npz(filedir + 'doctermat.npz')

# Setting the features and targets
X = tfidf_to_counts(np.array(doctermat.todense(),
                             dtype=np.uint16))
y = np.array(corpus.aucaseyn, dtype=np.uint8)
n_range = range(corpus.shape[0])

# Toggle for the optimization loop
optimize = False
opt_iter = 30

if optimize:
    # Regular function for hyperparameter evaluation
    def evaluate_hps(beta, C):    
        mod = NBSVM(C=C, beta=beta)
        mod.fit(X[train], y[train])
        guesses = mod.predict(X[val]).flatten()
        final_score = 1 - accuracy_score(y[val], guesses)
        params = np.array([beta, C])
        print('Params were ' + str(params))
        print('Error was ' + str(final_score) + '\n')
        return final_score

    # Bounds for the GP optimizer
    bounds = [{'name': 'beta',
               'type': 'continuous',
               'domain': (0.8, 1.0)},
              {'name': 'C',
               'type': 'discrete',
               'domain': (0.001, 0.01, 1.0, 2, 2**2)}
              ]
    
    # Function for GPyOpt to optimize
    def f(x):
        print(x)
        eval = evaluate_hps(beta=float(x[:, 0]),
                            C=float(x[:, 1]))
        return eval

    # Running the optimization
    train, val = train_test_split(n_range,
                                  test_size=0.3,
                                  stratify=y,
                                  random_state=10221983)
    opt_mod = GPyOpt.methods.BayesianOptimization(f=f,
                                                  num_cores=20,
                                                  domain=bounds,
                                                  initial_design_numdata=5)
    opt_mod.run_optimization(opt_iter)
    best = opt_mod.x_opt
    
    # Saving the best parameters to CSV
    pd.Series(best).to_csv(filedir + 'models/best_nbsvm_params.csv',
                           index=False)

# Running the splits
stats = pd.DataFrame(np.zeros([10, 15]))
for i, seed in enumerate(seeds):
    train, test = train_test_split(n_range,
                                  stratify=y,
                                  random_state=seed,
                                  test_size=0.3)
    
    if i == 0:
        test_guesses = pd.DataFrame(np.zeros([X[test].shape[0], 10]))
    
    # Fitting the model
    mod = simpleNBSVM(C=0.001)
    print('Fitting model ' + str(i))
    mod.fit(X[train], y[train])
    
    # Getting the predicted probs and thresholded guesses
    guesses = mod.predict(X[test]).flatten()
    test_guesses.iloc[:, i] = guesses
    bin_stats = binary_diagnostics(y[test], guesses, accuracy=True)
    print(bin_stats)
    stats.iloc[i, :] = bin_stats.values

# Writing the output to CSV
stats.columns = ['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'ppv', 'npv',
                 'f1', 'acc', 'true', 'pred', 'abs', 'rel', 'mcnemar']
stats.to_csv(filedir + 'stats/nbsvm_simple_stats.csv',
             index=False)
test_guesses.to_csv(filedir + 'guesses/nbsvm_simple_test_guesses.csv', 
                    index=False)
