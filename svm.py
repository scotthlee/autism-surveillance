import numpy as np
import pandas as pd
import GPy, GPyOpt
import h5py

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import brier_score_loss as brier_score
from scipy.sparse import load_npz

from metrics import binary_diagnostics, threshold, grid_metrics

# Importing the integer sequences and targets
indir = 'C:/data/addm/'
outdir = indir + 'models/'
corpus = pd.read_csv(indir + 'corpus_with_lemmas_clean.csv')
y = np.array(corpus.aucaseyn, dtype=np.uint8)
X = load_npz(indir + 'doctermat.npz')

'''
Optimizing the hyperparameters
'''
# Values for the fixed hyperparameters
seeds = np.array(pd.read_csv(indir + 'seeds.csv')).flatten()
n_range = range(corpus.shape[0])
opt_iter = 50

# Toggle for the optimization loop
optimize = False

if optimize:
    # Regular function for hyperparameter evaluation
    def evaluate_hps(C):    
        mod = LinearSVC(C=C)
        mod.fit(X[train], y[train])
        final_score = 1 - mod.score(X[test], y[test])
        print('C was ' + str(C))
        print('Accuracy was ' + str(final_score) + '\n')
        return final_score

    # Bounds for the GP optimizer
    bounds = [{'name': 'C',
               'type': 'discrete',
               'domain': (0.0001, 0.001, 0.01, 0.1, 1.0, 2.0, 5.0)}
              ]

    # Function for GPyOpt to optimize
    def f(x):
        print(x)
        eval = evaluate_hps(C=float(x[:, 0]))
        return eval

    # Running the optimization
    train, test = train_test_split(n_range,
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
    pd.Series(best).to_csv(outdir + 'best_svm_params.csv', index=False)

'''
Running the 10 splits with the optimized hyperparameters
'''
# Running the 10 splits with the optimized hyperparameters
# Setting up the array for the scores
stats = pd.DataFrame(np.zeros([10, 15]))
stats.columns = ['tp', 'fp', 'tn', 'fn', 'sens',
                 'spec', 'ppv', 'npv', 'f1', 'acc',
                 'true', 'pred', 'abs', 'rel', 'mcnemar']

# Loading previously-optimized hyperparameters
best = np.array(pd.read_csv(outdir + 'best_svm_params.csv',
                            header=None)).flatten()

for i, seed in enumerate(seeds):
    train, test = train_test_split(n_range,
                                   test_size=0.3,
                                   stratify=y,
                                   random_state=seed)
    
    if i == 0:
        test_guesses = pd.DataFrame(np.zeros([X[test].shape[0], 10]))
    
    # Training and testing the final model
    mod = LinearSVC(C=best[0])
    mod.fit(X[train], y[train])
    
    # Getting the predicted probs and thresholded guesses
    guesses = mod.predict(X[test])
    test_guesses.iloc[:, i] = guesses
    bin_stats = binary_diagnostics(y[test], guesses, accuracy=True)
    print(bin_stats)
    stats.iloc[i, :] = bin_stats.values

# Writing the stats file to CSV
test_guesses.to_csv(indir + 'svm_guesses.csv', index=False)
stats.to_csv(indir + 'svm_stats.csv', index=False)
