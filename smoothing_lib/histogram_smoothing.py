import numpy as np
import hist
from smoothing_lib.statistics_utils import (
    get_local_extrema_binning,
    get_ratio_hist,
    apply_smoothing_kernel
)

def smooth_histogram(hnom_hist, hsys_hist, nmax, apply_smooth=True):
    hnom = hnom_hist.values()
    hsys = hsys_hist.values()
    hnom_err = np.sqrt(hnom_hist.variances())

    bins = get_local_extrema_binning(hnom, hsys, hnom_err, nmax)
    ratio = get_ratio_hist(hnom, hsys, bins)

    if apply_smooth and len(ratio) > 2:
        ratio = apply_smoothing_kernel(ratio)

    smoothed = ratio * hnom

    norm_old, norm_new = np.sum(hsys), np.sum(smoothed)
    if norm_new > 1e-9:
        scale = norm_old / norm_new
        smoothed *= scale

    edges = hsys_hist.axes[0].edges
    hnew = hist.Hist(hist.axis.Regular(len(edges)-1, edges[0], edges[-1]))
    hnew[...] = smoothed
    return hnew

def smooth_rebin_monotonic(hnom_hist, hsys_hist):

    return smooth_histogram(hnom_hist, hsys_hist, nmax=0)

def smooth_rebin_parabolic(hnom_hist, hsys_hist):

    return smooth_histogram(hnom_hist, hsys_hist, nmax=1)
