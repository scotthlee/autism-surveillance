import numpy as np
import pandas as pd
import GPy, GPyOpt

from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from sklearn.metrics import brier_score_loss as brier_score
from scipy.sparse import load_npz
from keras import regularizers, optimizers

from models import slowText
from tools import limit_mem
from metrics import binary_diagnostics, threshold, grid_metrics

# Importing the integer sequences and targets
indir = 'C:/data/addm/'
outdir = indir + 'models/'
corpus = pd.read_csv(indir + 'corpus_with_lemmas_clean.csv')
targets = np.array(corpus.aucaseyn, dtype=np.uint8)
sparse_sents = load_npz(indir + 'unigram_doctermat.npz')

# Choosing a feature representation
X = np.array(sparse_sents.todense(), dtype=np.uint8)
vocab_size = X.shape[1]

# Reshaping the targets for Keras
y = targets

'''
Optimizing the hyperparameters
'''
# Values for the fixed hyperparameters
seeds = np.array(pd.read_csv(indir + 'seeds.csv')).flatten()
n_range = range(corpus.shape[0])
pred_batch = 1024
max_length = X.shape[1]
epochs = 10000
opt_iter = 20

# Toggle for the optimization loop
optimize = False

if optimize:
    # Regular function for hyperparameter evaluation
    def evaluate_hps(dropout,
                     patience,
                     e_size,
                     batch_size,
                     learning_rate):    
        stop = EarlyStopping(monitor='val_loss',
                             patience=patience)
        mod = slowText(vocab_size,
                       max_length,
                       embedding_size=e_size,
                       dropout=dropout)
        adam = optimizers.Adam(lr=learning_rate)
        mod.compile(optimizer=adam,
                    loss='binary_crossentropy')
        fit = mod.fit(X[train], y[train],
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=2,
                      callbacks=[stop],
                      validation_data=([X[val], y[val]]))
        final_score = np.mean(fit.history['val_loss'][-patience:])
        params = np.array([dropout, patience, e_size,
                           batch_size, learning_rate])
        print('Params were ' + str(params))
        print('Mean loss was ' + str(final_score) + '\n')
        return final_score

    # Bounds for the GP optimizer
    bounds = [{'name': 'dropout',
               'type': 'continuous',
               'domain': (0.0, 0.9)},
              {'name': 'patience',
               'type': 'discrete',
               'domain': (2, 5, 10)},
              {'name': 'e_size',
               'type': 'discrete',
               'domain': (64, 128, 256, 512)},
              {'name': 'batch_size',
               'type': 'discrete',
               'domain': (32, 64, 128, 256)},
              {'name': 'learning_rate',
               'type': 'discrete',
               'domain': (0.00001, 0.0001, 0.001)}
              ]

    # Function for GPyOpt to optimize
    def f(x):
        limit_mem()
        print(x)
        eval = evaluate_hps(dropout=float(x[:, 0]),
                            patience=int(x[:, 1]),
                            e_size=int(x[:, 2]),
                            batch_size=int(x[:, 3]),
                            learning_rate=float(x[:, 4]))
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
    pd.Series(best).to_csv(outdir + 'best_st_bg_params.csv', index=False)

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
best = np.array(pd.read_csv(outdir + 'best_st_bg_params.csv',
                            header=None)).flatten()

for i, seed in enumerate(seeds):
    not_test, test = train_test_split(n_range,
                                      test_size=0.3,
                                      stratify=targets,
                                      random_state=seed)
    train, val = train_test_split(not_test,
                                  test_size=0.2,
                                  stratify=targets[not_test],
                                  random_state=seed)
    
    if i == 0:
        test_guesses = pd.DataFrame(np.zeros([X[test].shape[0], 10]))
    
    # Training and testing the final model
    limit_mem()
    modfile = outdir + 'ft_mod.hdf5'
    stop = EarlyStopping(monitor='val_loss',
                         patience=int(best[1]))
    check = ModelCheckpoint(filepath=modfile,
                            save_best_only=True,
                            verbose=1)
    mod = slowText(vocab_size,
                   max_length,
                   embedding_size=int(best[2]),
                   dropout=best[0])
    adam = optimizers.Adam(lr=best[4])
    mod.compile(optimizer=adam,
                loss='binary_crossentropy')
    fit = mod.fit(X[train], y[train],
                  batch_size=int(best[3]),
                  epochs=epochs,
                  verbose=2,
                  callbacks=[stop],
                  validation_data=([X[val], y[val]]))
    #mod = load_model(modfile)
    
    # Getting the predicted probs and thresholded guesses
    val_probs = mod.predict(X[val]).flatten()
    val_gm = grid_metrics(y[val], val_probs)
    val_cut = val_gm.cutoff[np.argmax(val_gm.f1)]
    pos_probs = mod.predict(X[test]).flatten()
    guesses = threshold(pos_probs, val_cut)
    test_guesses.iloc[:, i] = guesses
    bin_stats = binary_diagnostics(y[test], guesses, accuracy=True)
    print(bin_stats)
    bs = brier_score(y[test], pos_probs)
    bin_stats['bs'] = bs
    stats.iloc[i, :] = bin_stats.values

# Writing the stats file to CSV
#stats.to_csv(indir + 'st_bg_f1_stats.csv', index=False)
test_guesses.to_csv(indir + 'st_test_guesses.csv', index=False)
