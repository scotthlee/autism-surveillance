import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.metrics import brier_score_loss as brier_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.sparse import load_npz

from stuff.metrics import binary_diagnostics, grid_metrics, threshold

# Importing the data
filedir = 'C:/data/addm/'
seeds = np.array(pd.read_csv(filedir + 'seeds.csv')).flatten()
corpus = pd.read_csv(filedir + 'corpus_with_lemmas_clean.csv')
y = np.array(corpus.aucaseyn, dtype=np.uint8)
doctermat = load_npz(filedir + 'doctermat.npz')
X = doctermat

# Optional TF-IDF transformation
tf = TfidfTransformer()
X = tf.fit_transform(doctermat)

# Options for the training loops
optimize = False
n_range = range(X.shape[0])
top_ns = np.arange(10, 200, step=10)
scores = np.zeros(len(top_ns))
trees = 1000

# Hyper-hyperparameters
#pruning_method = 'RFE'
pruning_method = 'simple'

if optimize:
    train, val = train_test_split(n_range,
                                  test_size=0.3,
                                  stratify=y,
                                  random_state=10221983)
    # This loop finds the best number of features to select
    for i, top_n in enumerate(top_ns):    
        mod = RandomForestClassifier(n_estimators=trees,
                                     n_jobs=-1)
        mod.fit(X[train], y[train])
        imps = np.argsort(mod.feature_importances_)
        if pruning_method == 'simple':
            X_trimmed = X[:, imps[-top_n:]]
            trim_mod = RandomForestClassifier(n_estimators=trees,
                                              n_jobs=-1)
            trim_mod.fit(X_trimmed[train], y[train])
            preds = trim_mod.predict(X_trimmed[val])
            final_score = f1_score(y[val], preds)                
        else:
            imps = np.argsort(mod.feature_importances_)
            X_trimmed = X[:, imps[-250:]]
            rfe = RFE(mod,
                      verbose=1,
                      n_features_to_select=top_n,
                      step=10)
            rfe.fit(X_trimmed[train], y[train])
            print(rfe.n_features_)
            final_score = f1_score(y[val],
                                   rfe.predict(X_trimmed[val]))
        scores[i] = final_score
        print('Top n was ' + str(top_n))
        print('Accuracy was ' + str(final_score) + '\n')
    
    best = pd.DataFrame([top_ns, scores]).transpose()
    best.columns = ['top_n', 'score']
    
    # Saving the best parameters to CSV
    if pruning_method == 'simple':
        best.to_csv(filedir + 'models/rf_simple_topn.csv', index=False)
    elif pruning_method == 'RFE':
        best.to_csv(filedir + 'models/rf_rfe_topn.csv', index=False)

'''
Running the 10 train-test splits
'''
# Importing the best number of features
if pruning_method == 'simple':
    best = pd.read_csv(filedir + 'models/rf_simple_topn.csv')
    top_n = best.top_n[np.argmax(best.score)]
elif pruning_method == 'RFE':
    best = pd.read_csv(filedir + 'models/rf_rfe_topn.csv')
    top_n = best.top_n[11]

# Running the splits
n_range = range(corpus.shape[0])
stats = pd.DataFrame(np.zeros([10, 16]))
for i, seed in enumerate(seeds):
    train, test = train_test_split(n_range,
                                   stratify=y,
                                   random_state=seed,
                                   test_size=0.3)
    
    # Making a holder for the test predictions
    if i == 0:
        test_guesses = pd.DataFrame(np.zeros([X[test].shape[0], 10]))
    
    # Fitting the model
    mod = RandomForestClassifier(n_estimators=trees, n_jobs=-1)
    print('Fitting model ' + str(i))
    mod.fit(X[train], y[train])
    imps = np.argsort(mod.feature_importances_)
    
    if pruning_method == 'simple':
        X_trim = X[:, imps[-top_n:]]
        final_mod = RandomForestClassifier(n_estimators=trees, n_jobs=-1)
        final_mod.fit(X_trim[train], y[train])
    
    elif pruning_method == 'RFE':
        X_trim = X[:, imps[-250:]]
        final_mod = RFE(mod,
                        verbose=1,
                        n_features_to_select=top_n,
                        step=10)
        final_mod.fit(X_trim[train], y[train])
    
    # Getting the predicted probs and thresholded guesses
    pos_probs = final_mod.predict_proba(X_trim[test])[:, 1].flatten()
    guesses = final_mod.predict(X_trim[test])
    test_guesses.iloc[:, i] = guesses
    bin_stats = binary_diagnostics(y[test], guesses, accuracy=True)
    print(bin_stats)
    bs = brier_score(y[test], pos_probs)
    bin_stats['bs'] = bs
    stats.iloc[i, :] = bin_stats.values

# Writing the output to CSV
stats.columns = ['tp', 'fp', 'tn', 'fn', 'sens', 'spec', 'ppv', 'npv',
                 'f1', 'acc', 'true', 'pred', 'abs', 'rel', 'mcnemar', 'brier']

if pruning_method == 'simple':
    #stats.to_csv(filedir + 'rf_simple_stats.csv', index=False)
    test_guesses.to_csv(filedir + 'rf_simple_test_guesses.csv', index=False)
elif pruning_method == 'RFE':
    #stats.to_csv(filedir + 'rf_rfe_stats.csv', index=False)
    test_guesses.to_csv(filedir + 'rf_rfe_test_guesses.csv', index=False)

