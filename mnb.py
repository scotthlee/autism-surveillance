import numpy as np
import pandas as pd
import GPy, GPyOpt
import h5py

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import brier_score_loss as brier_score
from scipy.sparse import load_npz

from stuff.metrics import binary_diagnostics, threshold, grid_metrics

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
    def evaluate_hps(alpha):    
        mod = MultinomialNB(alpha=alpha)
        mod.fit(X[train], y[train])
        final_score = brier_score(y[test], mod.predict_proba(X[test])[:, 1])
        print('Alpha was ' + str(alpha))
        print('Accuracy was ' + str(final_score) + '\n')
        return final_score

    # Bounds for the GP optimizer
    bounds = [{'name': 'alpha',
               'type': 'continuous',
               'domain': (0.0001, 1.0)}
              ]

    # Function for GPyOpt to optimize
    def f(x):
        print(x)
        eval = evaluate_hps(alpha=float(x[:, 0]))
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
    pd.Series(best).to_csv(outdir + 'best_mnb_params.csv', index=False)

'''
Running the 10 splits with the optimized hyperparameters
'''
# Running the 10 splits with the optimized hyperparameters
# Setting up the array for the scores
stats = pd.DataFrame(np.zeros([10, 16]))
stats.columns = ['tp', 'fp', 'tn', 'fn', 'sens',
                 'spec', 'ppv', 'npv', 'f1', 'acc',
                 'true', 'pred', 'abs', 'rel',
                 'mcnemar', 'brier']

# Loading previously-optimized hyperparameters
best = np.array(pd.read_csv(outdir + 'best_mnb_params.csv',
                            header=None)).flatten()

for i, seed in enumerate(seeds):
    train, test = train_test_split(n_range,
                                   test_size=0.3,
                                   stratify=y,
                                   random_state=seed)
    
    if i == 0:
        test_guesses = pd.DataFrame(np.zeros([X[test].shape[0], 10]))
    
    # Training and testing the final model
    mod = MultinomialNB(alpha=0.03)
    mod.fit(X[train], y[train])
    
    # Getting the predicted probs and thresholded guesses
    pos_probs = mod.predict_proba(X[test])[:, 1]
    guesses = mod.predict(X[test])
    test_guesses.iloc[:, i] = guesses
    bin_stats = binary_diagnostics(y[test], guesses, accuracy=True)
    print(bin_stats)
    bs = brier_score(y[test], pos_probs)
    bin_stats['bs'] = bs
    stats.iloc[i, :] = bin_stats.values

# Writing the stats file to CSV
test_guesses.to_csv(indir + 'mnb_guesses.csv', index=False)
stats.to_csv(indir + 'mnb_stats.csv', index=False)
