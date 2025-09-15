import numpy as np
from scipy.stats import ks_2samp, chi2
from typing import List


def reduced_chi2(data1: np.ndarray, data2: np.ndarray, variance: np.ndarray) -> float:
    """
    Compute the reduced chi-squared p-value between two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        First dataset (observed values).
    data2 : np.ndarray
        Second dataset (expected or model values).
    variance : np.ndarray
        Variances associated with data points.

    Returns
    -------
    float
        p-value from the chi-squared test. Returns 1.0 if the calculation is not possible.
    """
    mask = variance > 0
    if not np.any(mask):
        return 1.0
    chi_2 = np.sum(((data1[mask] - data2[mask]) ** 2) / variance[mask])
    n_dof = np.count_nonzero(mask) - 1
    if n_dof <= 0:
        return 1.0
    return chi2.sf(chi_2, n_dof)


def stat_error(values: np.ndarray, errors: np.ndarray, beg: int, end: int) -> float:
    """
    Compute the relative statistical error in a given interval.

    Parameters
    ----------
    values : np.ndarray
        Array of bin values (counts).
    errors : np.ndarray
        Array of bin errors.
    beg : int
        Start index (inclusive).
    end : int
        End index (exclusive).

    Returns
    -------
    float
        Relative statistical error (sigma / integral).
        Returns np.inf if the integral is zero.
    """
    integral = np.sum(values[beg:end])
    err2 = np.sum(errors[beg:end] ** 2)
    if integral == 0:
        return np.inf
    return np.sqrt(err2) / integral


def compute_chi2(hnom: np.ndarray, hsys: np.ndarray, hnom_err: np.ndarray, beg: int, end: int) -> float:
    """
    Compute chi-squared statistic for ratio consistency in a given interval.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram values.
    hsys : np.ndarray
        Systematic histogram values.
    hnom_err : np.ndarray
        Errors for the nominal histogram.
    beg : int
        Start index (inclusive).
    end : int
        End index (inclusive).

    Returns
    -------
    float
        Chi-squared value.
    """
    nom_int = np.sum(hnom[beg:end + 1])
    sys_int = np.sum(hsys[beg:end + 1])
    ratio = sys_int / nom_int if nom_int > 0 else 0.0
    chi2_val = 0.0
    for i in range(beg, end + 1):
        if hnom[i] != 0:
            iratio = hsys[i] / hnom[i]
            err = hnom_err[i] / hnom[i] if hnom[i] > 0 else 1.0
            chi2_val += ((iratio - ratio) / err) ** 2
    return chi2_val


def find_smaller_chi2(hnom: np.ndarray, hsys: np.ndarray, hnom_err: np.ndarray, extrema: List[int]) -> int:
    """
    Find the interval with the smallest chi-squared among extrema.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram values.
    hsys : np.ndarray
        Systematic histogram values.
    hnom_err : np.ndarray
        Errors for the nominal histogram.
    extrema : list of int
        List of extrema positions.

    Returns
    -------
    int
        Index of the extremum interval with the smallest chi-squared value.
    """
    minval, pos = 1e9, 0
    for i in range(len(extrema) - 1):
        chi2_val = compute_chi2(hnom, hsys, hnom_err, extrema[i], extrema[i + 1])
        if chi2_val < minval:
            minval, pos = chi2_val, i
    return pos


def merge_bins(lo: int, hi: int, bins: List[int]) -> List[int]:
    """
    Merge bins between two boundaries (inclusive).

    Parameters
    ----------
    lo : int
        Lower boundary.
    hi : int
        Upper boundary.
    bins : list of int
        Current bin edges.

    Returns
    -------
    list of int
        Updated list of bins with merged intervals.
    """
    to_remove = [i for i, b in enumerate(bins) if lo < b < hi + 1]
    for i in reversed(to_remove):
        bins.pop(i)
    return bins


def get_ratio_hist(hnom: np.ndarray, hsys: np.ndarray, bins: List[int]) -> np.ndarray:
    """
    Compute ratio histogram between nominal and systematic histograms.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram values.
    hsys : np.ndarray
        Systematic histogram values.
    bins : list of int
        Bin edges.

    Returns
    -------
    np.ndarray
        Ratio histogram values.
    """
    ratio = np.zeros_like(hnom, dtype=float)
    for i in range(len(bins) - 1):
        beg, end = bins[i], bins[i + 1]
        nom_int = np.sum(hnom[beg:end])
        sys_int = np.sum(hsys[beg:end])
        r = sys_int / nom_int if nom_int > 0 else 0.0
        ratio[beg:end] = r
    return ratio


def find_extrema(values: np.ndarray, tol: float = 1e-6) -> List[int]:
    """
    Find local extrema in a sequence of values.

    Parameters
    ----------
    values : np.ndarray
        Input array of values.
    tol : float, optional
        Tolerance threshold for detecting changes, by default 1e-6.

    Returns
    -------
    list of int
        List of extrema indices (including 0 and last index).
    """
    extrema = [0]
    status, k = 0, 0
    for i in range(1, len(values)):
        if values[i] < tol:
            continue
        if status == 1 and values[i] < values[k] - tol:
            extrema.append(i - 1)
            status = -1
        elif status == -1 and values[i] > values[k] + tol:
            extrema.append(i - 1)
            status = 1
        elif status == 0:
            if values[i] < values[k] - tol:
                status = -1
            elif values[i] > values[k] + tol:
                status = 1
        k = i
    extrema.append(len(values) - 1)
    return sorted(set(extrema))


def apply_smoothing_kernel(values: np.ndarray) -> np.ndarray:
    """
    Apply a smoothing kernel to values.

    Parameters
    ----------
    values : np.ndarray
        Input values.

    Returns
    -------
    np.ndarray
        Smoothed values.
    """
    if len(values) <= 2:
        return values.copy()
    smooth = values.copy()
    smooth[1:-1] = (2 * values[1:-1] + values[:-2] + values[2:]) / 4.0
    return smooth


def get_local_extrema_binning(
    hnom: np.ndarray,
    hsys: np.ndarray,
    hnom_err: np.ndarray,
    nmax: int,
    stat_err_threshold: float = 0.05,
) -> List[int]:
    """
    Determine optimal binning based on local extrema and statistical error.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram values.
    hsys : np.ndarray
        Systematic histogram values.
    hnom_err : np.ndarray
        Errors for nominal histogram.
    nmax : int
        Maximum number of extrema allowed.
    stat_err_threshold : float, optional
        Statistical error threshold, by default 0.05.

    Returns
    -------
    list of int
        Optimal bin edges.
    """
    n_bins = len(hnom)

    total_sum = np.sum(hnom)
    total_err = np.sqrt(np.sum(hnom_err ** 2))
    if total_sum > 0 and abs(total_err / total_sum) > stat_err_threshold:
        return [0, n_bins]

    bins = list(range(n_bins + 1))
    ratio = get_ratio_hist(hnom, hsys, bins)
    extrema = find_extrema(ratio)

    while len(extrema) > nmax + 2:
        pos = find_smaller_chi2(hnom, hsys, hnom_err, extrema)
        bins = merge_bins(extrema[pos], extrema[pos + 1], bins)
        ratio = get_ratio_hist(hnom, hsys, bins)
        extrema = find_extrema(ratio)

    bins = _second_pass_bins(hnom, hnom_err, bins, stat_err_threshold)
    return bins


def _second_pass_bins(
    hnom: np.ndarray, hnom_err: np.ndarray, bins: List[int], stat_err_threshold: float
) -> List[int]:
    """
    Perform a second pass to remove bins with large statistical errors.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram values.
    hnom_err : np.ndarray
        Errors for the nominal histogram.
    bins : list of int
        Current bin edges.
    stat_err_threshold : float
        Statistical error threshold.

    Returns
    -------
    list of int
        Updated list of bins after filtering.
    """
    fst_idx = len(bins) - 1
    lst_idx = len(bins) - 1
    to_remove = []
    while fst_idx != 0:
        if fst_idx == lst_idx:
            fst_idx -= 1
        else:
            beg, end = bins[fst_idx], bins[lst_idx]
            se = stat_error(hnom, hnom_err, beg, end)
            if se > stat_err_threshold or np.isnan(se):
                to_remove.append(fst_idx)
                fst_idx -= 1
            else:
                lst_idx = fst_idx
    for i in to_remove:
        del bins[i]
    return bins

