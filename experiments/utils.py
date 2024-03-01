import numpy as np

from statsmodels.stats.multitest import multipletests


def compute_significance(request_history, alpha):
    pvals = [v[0] for v in request_history.values()]
    rejects = [v[1] for v in request_history.values()]
    ground_truth_reject, ground_truth_pvals, _, ground_truth_alphacBonf = multipletests(pvals, alpha=alpha, method='bonferroni')
    num_rejects_ground_truth = sum([1 if reject else 0 for reject in ground_truth_reject])
    num_rejects = sum([1 if reject else 0 for reject in rejects])
    true_positives = sum([1 if reject and reject_ground_truth else 0 for reject, reject_ground_truth in zip(rejects, ground_truth_reject)])
    false_positives = sum([1 if reject and not reject_ground_truth else 0 for reject, reject_ground_truth in zip(rejects, ground_truth_reject)])

    if num_rejects_ground_truth > 0:
        power = true_positives / num_rejects_ground_truth
    else:
        power = np.nan
    
    if num_rejects > 0:
        fdr = false_positives / num_rejects
    else:
        fdr = np.nan

    return power, fdr
