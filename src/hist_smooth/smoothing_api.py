from hist_smooth.base_smoothing import (
    moving_median_odd,
    hanning,
    quadratic_interpolation,
    twicing,
)

from hist_smooth.histogram_smoothing import (
    smooth_histogram
)

from hist_smooth.tukey_smoothing import (
    ALGORITHMS
)

def tukey_smoothing(hnom, algorithm, endrule='median', twice=0, var_function = lambda histogram: histogram):

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
        fn = ALGORITHMS[algorithm]
    else:
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
            
    data = hnom.view().value
    data_var = hnom.view().variance

    mod_data = fn(data)

    if twice > 0:
        mod_data = twicing(data, mod_data, fn, times=twice)

    hnom.view().value = mod_data  
    hnom.view().variance = var_function(data_var)
    return hnom

def smooth_rebin_monotonic(hnom_hist, hsys_hist):

    return smooth_histogram(hnom_hist, hsys_hist, nmax=0)

def smooth_rebin_parabolic(hnom_hist, hsys_hist):

    return smooth_histogram(hnom_hist, hsys_hist, nmax=1)

def smooth_hist_general(hnom_hist, hsys_hist, algorithm,
                         apply_smooth=True, endrule='median',
                         twice=0):
    if algorithm == "monotonic":
        hnew = smooth_rebin_monotonic(hnom_hist, hsys_hist)
    elif algorithm == "parabolic":
	    hnew = smooth_rebin_parabolic(hnom_hist, hsys_hist)
    else:
        hnew = tukey_smoothing(hnom_hist, algorithm, endrule=endrule, twice=twice)
    return hnew
