import numpy as np
from scipy.stats import ks_2samp, chi2
import hist

def reduced_chi2(data1, data2, variance):

    mask = variance > 0
    if not np.any(mask):
        return 1.0
    chi_2 = np.sum(((data1[mask] - data2[mask]) ** 2) / (variance[mask]))
    n_dof = np.count_nonzero(mask) - 1
    if n_dof <= 0:
        return 1.0
    reduced_chi2_p_value = chi2.sf(chi_2, n_dof)
    return reduced_chi2_p_value

def stat_error(values, errors, beg, end):

    integral = np.sum(values[beg:end])
    err2 = np.sum(errors[beg:end]**2)
    if integral == 0:
        return np.inf
    return np.sqrt(err2) / integral

def compute_chi2(hnom, hsys, hnom_err, beg, end):

    nom_int = np.sum(hnom[beg:end+1])
    sys_int = np.sum(hsys[beg:end+1])
    ratio = sys_int / nom_int if nom_int > 0 else 0.0
    chi2_val = 0.0
    for i in range(beg, end + 1):
        if hnom[i] != 0:
            iratio = hsys[i] / hnom[i]
            err = hnom_err[i] / hnom[i] if hnom[i] > 0 else 1.0
            chi2_val += ((iratio - ratio) / err) ** 2
    return chi2_val

def find_smaller_chi2(hnom, hsys, hnom_err, extrema):

    minval, pos = 1e9, 0
    for i in range(len(extrema) - 1):
        chi2_val = compute_chi2(hnom, hsys, hnom_err, extrema[i], extrema[i+1])
        if chi2_val < minval:
            minval, pos = chi2_val, i
    return pos

def merge_bins(lo, hi, bins):

    to_remove = [i for i, b in enumerate(bins) if lo < b < hi+1]
    for i in reversed(to_remove):
        bins.pop(i)
    return bins

def get_ratio_hist(hnom, hsys, bins):

    ratio = np.zeros_like(hnom, dtype=float)
    for i in range(len(bins)-1):
        beg, end = bins[i], bins[i+1]
        nom_int = np.sum(hnom[beg:end])
        sys_int = np.sum(hsys[beg:end])
        r = sys_int/nom_int if nom_int > 0 else 0.0
        ratio[beg:end] = r
    return ratio

def find_extrema(values, tol=1e-6):

    extrema = [0]
    status, k = 0, 0 
    for i in range(1, len(values)):
        if values[i] < tol:
            continue
        if status == 1 and values[i] < values[k] - tol:
            extrema.append(i-1); status = -1
        elif status == -1 and values[i] > values[k] + tol:
            extrema.append(i-1); status = 1
        elif status == 0:
            if values[i] < values[k] - tol: status = -1
            elif values[i] > values[k] + tol: status = 1
        k = i
    extrema.append(len(values)-1)
    return sorted(list(set(extrema)))

def apply_smoothing_kernel(values):

    if len(values) <= 2: return values.copy()
    smooth = values.copy()
    smooth[1:-1] = (2*values[1:-1] + values[:-2] + values[2:]) / 4.0
    return smooth

def get_local_extrema_binning(hnom, hsys, hnom_err, nmax, stat_err_threshold=0.05):

    n_bins = len(hnom)
    total_sum, total_err = np.sum(hnom), np.sqrt(np.sum(hnom_err**2))
    if total_sum > 0 and abs(total_err/total_sum) > stat_err_threshold:
        return [0, n_bins]

    bins = list(range(n_bins + 1))
    ratio = get_ratio_hist(hnom, hsys, bins)
    extrema = find_extrema(ratio)

    while len(extrema) > nmax + 2:
        pos = find_smaller_chi2(hnom, hsys, hnom_err, extrema)
        bins = merge_bins(extrema[pos], extrema[pos+1], bins)
        ratio = get_ratio_hist(hnom, hsys, bins)
        extrema = find_extrema(ratio)

    to_remove = []
    for i in range(1, len(bins)):
        if stat_error(hnom, hnom_err, bins[i-1], bins[i]) > stat_err_threshold:
            to_remove.append(i)
    for idx in reversed(to_remove):
        bins.pop(idx)
    return bins
