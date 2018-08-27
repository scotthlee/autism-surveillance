import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score
from scipy.stats import chi2

# Quick function for thresholding probabilities
def threshold(probs, cutoff=.5):
    return np.array(probs >= cutoff).astype(np.uint8)

# Calculates McNemar's chi-squared statistic
def mcnemar_test(true, pred, cc=False):
    confmat = confusion_matrix(true, pred)
    b = int(confmat[0, 1])
    c = int(confmat[1, 0])
    if cc:
        stat = (abs(b - c) - 1)**2 / (b + c)
    else:
        stat = (b - c)**2 / (b + c)
    p = 1 - chi2(df=1).cdf(stat)
    outmat = np.array([b, c, stat, p]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['b', 'c', 'stat', 'pval'])
    return out

# Runs basic diagnostic stats on binary (only) predictions
def binary_diagnostics(true, pred, accuracy=False, counts=True):
    confmat = confusion_matrix(true, pred)
    tp = confmat[1, 1]
    fp = confmat[0, 1]
    tn = confmat[0, 0]
    fn = confmat[1, 0]
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    f1 = 2 * (sens * ppv) / (sens + ppv)
    outmat = np.array([tp, fp, tn, fn, sens,
                       spec, ppv, npv, f1]).reshape(-1, 1)
    out = pd.DataFrame(outmat.transpose(),
                       columns=['tp', 'fp', 'tn', 'fn', 'sens',
                                'spec', 'ppv', 'npv', 'f1'])
    if accuracy:
        out['acc'] = accuracy_score(true, pred)
    if counts:
        true_prev = int(np.sum(true == 1))
        pred_prev = int(np.sum(pred == 1))
        abs_diff = (true_prev - pred_prev) * -1
        rel_diff = abs_diff / true_prev
        mcnemar = mcnemar_test(true, pred).pval
        count_outmat = np.array([true_prev, pred_prev,
                                  abs_diff, rel_diff, mcnemar]).reshape(-1, 1)
        count_out = pd.DataFrame(count_outmat.transpose(),
                                 columns=['true', 'pred', 'abs', 'rel',
                                          'mcnemar'])
        out = pd.concat([out, count_out], axis=1)
    return out

# Runs basic diagnostic stats on binary or multiclass predictions
def diagnostics(true, pred, average='binary', accuracy=False, counts=False):
    sens = recall_score(true, pred, average=average)
    ppv = precision_score(true, pred, average=average)
    f1 = f1_score(true, pred, average=average)
    out = pd.DataFrame([sens, ppv, f1]).transpose()
    out.columns = ['sens', 'ppv', 'f1']
    if accuracy:
        out['acc'] = accuracy_score(true, pred)
    if counts:
        out['true'] = int(np.sum(true == 1))
        out['pred'] = int(np.sum(pred == 1))
        out['abs_diff'] = (out.true - out.pred) * -1
        out['rel_diff'] = out.abs_diff / out.true
    return out

# Finds the optimal threshold for a classifier based on a metric
def grid_metrics(targets,
                 guesses,
                 step=.01,
                 min=0.0,
                 max=1.0,
                 by='f1',
                 average='binary',
                 counts=True):
    cutoffs = np.arange(min, max, step)
    if len((guesses.shape)) == 2:
        guesses = guesses[:, 1]
    if average == 'binary':
        scores = pd.DataFrame(np.zeros(shape=(int(1/step), 15)),
                              columns=['cutoff', 'tp', 'fp', 'tn', 'fn',
                                       'sens', 'spec', 'ppv', 'npv', 'f1',
                                       'true', 'pred', 'abs', 'rel',
                                       'mcnemar'])
        for i, cutoff in enumerate(cutoffs):
            threshed = threshold(guesses, cutoff)
            stats = binary_diagnostics(targets, threshed)
            scores.iloc[i, 1:] = stats.values
            scores.cutoff[i] = cutoff
    else:
        scores = pd.DataFrame(np.zeros(shape=(int(1/step), 4)),
                              columns=['cutoff', 'sens', 'ppv', 'f1'])
        if counts:
            new = pd.DataFrame(np.zeros(shape=(int(1/step), 4)),
                                  columns=['true', 'pred',
                                           'abs_diff', 'rel_diff'])
            scores = pd.concat([scores, new], axis=1)
        for i, cutoff in enumerate(cutoffs):
            threshed = threshold(guesses, cutoff)
            stats = diagnostics(targets,
                                threshed,
                                average=average,
                                counts=counts)
            scores.iloc[i, 1:] = stats.values
            scores.cutoff[i] = cutoff
    return scores

# Calculates bootstrap confidence intervals for an estimator
def boot_cis(targets, guesses, n=100, a=0.05, average='binary', seed=10221983):
    colnames = ['sens', 'ppv', 'f1', 'true',
                'pred', 'abs_diff', 'rel_diff']
    scores = pd.DataFrame(np.zeros(shape=(n, 7)),
                          columns=colnames)
    np.random.seed(seed)
    seeds = np.random.randint(0, 1e6, n)
    for i in range(n):
        m = targets.shape[0]
        np.random.seed(seeds[i])
        boot = np.random.choice(range(m), size=m, replace=True)
        stats = diagnostics(targets[boot],
                            guesses[boot],
                            counts=True,
                            average=average)
        scores.iloc[i, :] = stats.values
    lower = (a / 2) * 100
    upper = 100 - lower
    cis = np.percentile(scores, q=(lower, upper), axis=0)
    stat = diagnostics(targets, guesses, counts=True).transpose()
    cis = pd.DataFrame(cis.transpose(),
                       columns=['lower', 'upper'],
                       index=colnames)
    cis = pd.concat([stat, cis], axis=1)
    cis.columns = ['stat', 'lower', 'upper']
    return {'cis':cis, 'scores':scores}

# Takes two boot_cis() objects and returns the difference between them
def diff_boot_cis(ref, comp, n=100, a=0.05, reverse=True, diff_type=None):
    ref_stat = ref['cis']['stat']
    comp_stat = comp['cis']['stat']
    if reverse:
        diffs = comp['scores'] - ref['scores']
        stat = comp_stat - ref_stat
    else:
        diffs = ref['scores'] - comp['scores']
        stat = ref_stat - comp_stat
    if diff_type == 'relative':
        diffs = diffs / ref['scores']
        stat = stat / ref_stat
    lower = (a / 2) * 100
    upper = 100 - lower
    cis = np.percentile(diffs, q=(lower, upper), axis=0)
    cis = pd.DataFrame(cis.transpose(),
                       columns=['lower', 'upper'],
                       index=ref['scores'].columns)
    cis = pd.concat([ref_stat, comp_stat, stat, cis], axis=1)
    cis.columns = ['ref', 'comp', 'diff', 'lower', 'upper']
    if diff_type == 'relative':
        cis = cis.drop(['true', 'pred', 'abs_diff', 'rel_diff'], axis=0)
    return cis
