import numpy as np
import boost_histogram as bh
import hist

from hist_smooth.base_smoothing import (
    moving_median_odd,
    hanning,
    quadratic_interpolation,
    twicing,
)

from hist_smooth.tukey_smoothing import (
    ALGORITHMS
)

from hist_smooth.statistics_utils import (
    get_local_extrema_binning,
    get_ratio_hist,
    apply_smoothing_kernel
)

from hist_smooth.base_smoothing import (
    moving_median_odd, hanning, even_median_four, 
    even_median_two, quadratic_interpolation
)


def smh_4253H(data, endrule='median', **kwargs):
    result = even_median_four(data)
    result = even_median_two(result)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_353QH(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 3, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_3G53QH(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 3, endrule)
    result = hanning(result, condition=True)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_53H(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_95H(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 9, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = hanning(result)
    return result

ALGORITHMS = {
    '4253H': smh_4253H,
    '353QH': smh_353QH,
    '3G53QH': smh_3G53QH,
    '53H': smh_53H,
    '95H': smh_95H,
}

def build_tuckey_fn(algorithm, endrule='median'):
    def wrap_median(prev_fn, k):
        return lambda d: prev_fn(moving_median_odd(d, k, endrule))

    def wrap_hanning(prev_fn):
        return lambda d: prev_fn(hanning(d))

    def wrap_gaussian_hanning(prev_fn):
        return lambda d: prev_fn(hanning(d, True))

    def wrap_quadratic(prev_fn):
        return lambda d: prev_fn(quadratic_interpolation(d))

    action_wrappers = {
        'H': wrap_hanning,
        'G': wrap_gaussian_hanning,
        'Q': wrap_quadratic
    }

    if algorithm in ALGORITHMS:
        return ALGORITHMS[algorithm]

    fn = lambda d: d
    for action in reversed(algorithm):
        if action.isdigit():
            fn = wrap_median(fn, int(action))
        elif action.isalpha():
            if action not in action_wrappers:
                raise ValueError(f"Invalid action '{action}'")
            fn = action_wrappers[action](fn)
        else:
            raise ValueError(f"Invalid action character '{action}'")
    return fn

def tukey_smoothing(data, algorithm, endrule='median', twice=0, var_function=lambda h: h):

    fn = build_tuckey_fn(algorithm, endrule)

    mod_data = fn(data)

    if twice > 0:
        mod_data = twicing(data, mod_data, fn, times=twice)

    return mod_data

def boost_hist_tukey(hnom, algorithm, endrule='median', twice=0, var_function=lambda h: h):

    num_of_dim = hnom.view().ndim
    if num_of_dim == 1:
        data = hnom.values()
    else:
        data = hnom.view().value
        data_var = hnom.view().variance

    mod_data = tukey_smoothing(data=data, algorithm=algorithm, endrule=endrule, twice=twice, var_function=var_function)
    
    if num_of_dim == 1:
        hnom[...] = mod_data 
    else:
        hnom.view().value = mod_data  
        hnom.view().variance = var_function(data_var)

    return hnom

def smooth_rebin_monotonic(hnom_hist, hsys_hist):

    return boost_hist_rebin(hnom_hist, hsys_hist, nmax=0)

def smooth_rebin_parabolic(hnom_hist, hsys_hist):

    return boost_hist_rebin(hnom_hist, hsys_hist, nmax=1)

def smooth_hist_general(hnom_hist, algorithm, hsys_hist=None,
                         apply_smooth=True, endrule='median',
                         twice=0):
    hnom_hist = hnom_hist.copy()
    
    if algorithm == "monotonic":
        hnew = smooth_rebin_monotonic(hnom_hist, hsys_hist)
    elif algorithm == "parabolic":
	    hnew = smooth_rebin_parabolic(hnom_hist, hsys_hist)
    else:
        hnew = boost_hist_tukey(hnom_hist, algorithm, endrule=endrule, twice=twice)
    return hnew

def hist_rebin(hnom, hsys, hnom_err, nmax, apply_smooth=True):

    bins = get_local_extrema_binning(hnom, hsys, hnom_err, nmax)
    ratio = get_ratio_hist(hnom, hsys, bins)

    if apply_smooth and len(ratio) > 2:
        ratio = apply_smoothing_kernel(ratio)

    smoothed = ratio * hnom

    norm_old, norm_new = np.sum(hsys), np.sum(smoothed)
    if norm_new > 1e-9:
        scale = norm_old / norm_new
        smoothed *= scale

    return smoothed

def boost_hist_rebin(hnom_hist, hsys_hist, nmax, apply_smooth=True):
    hnom = hnom_hist.values()
    hsys = hsys_hist.values()
    hnom_err = np.sqrt(hnom_hist.variances())

    smoothed = hist_rebin(hnom=hnom, hsys=hsys, hnom_err=hnom_err, nmax=nmax, apply_smooth=apply_smooth)

    hnom[...] = [(i, 0) for i in smoothed]
    return hnom

