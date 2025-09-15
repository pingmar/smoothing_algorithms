import numpy as np
from typing import Callable
import hist

from hist_smooth.base_smoothing import (
    moving_median_odd,
    hanning,
    quadratic_interpolation,
    twicing,
    even_median_four,
    even_median_two,
)
from hist_smooth.statistics_utils import (
    get_local_extrema_binning,
    get_ratio_hist,
    apply_smoothing_kernel,
)


def smh_4253H(data: np.ndarray, endrule: str = "median", **kwargs) -> np.ndarray:
    """
    Tukey's 4253H smoothing procedure.

    Steps:
        1. 4-point moving median
        2. 2-point moving median
        3. 5-point moving median
        4. 3-point moving median
        5. Hanning smoothing

    Parameters
    ----------
    data : np.ndarray
        Input 1D numeric sequence.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    np.ndarray
        Smoothed sequence of same length as input.
    """
    result = even_median_four(data)
    result = even_median_two(result)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_353QH(data: np.ndarray, endrule: str = "median", **kwargs) -> np.ndarray:
    """
    Tukey's 353QH smoothing procedure.

    Steps:
        1. 3-point moving median
        2. 5-point moving median
        3. 3-point moving median
        4. Quadratic interpolation
        5. Hanning smoothing

    Parameters
    ----------
    data : np.ndarray
        Input 1D numeric sequence.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    np.ndarray
        Smoothed sequence.
    """
    result = moving_median_odd(data, 3, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_3G53QH(data: np.ndarray, endrule: str = "median", **kwargs) -> np.ndarray:
    """
    Tukey's 3G53QH smoothing procedure.

    Steps:
        1. 3-point moving median
        2. Conditional Hanning smoothing
        3. 5-point moving median
        4. 3-point moving median
        5. Quadratic interpolation
        6. Hanning smoothing

    Parameters
    ----------
    data : np.ndarray
        Input 1D numeric sequence.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    np.ndarray
        Smoothed sequence.
    """
    result = moving_median_odd(data, 3, endrule)
    result = hanning(result, condition=True)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_53H(data: np.ndarray, endrule: str = "median", **kwargs) -> np.ndarray:
    """
    Tukey's 53H smoothing procedure.

    Steps:
        1. 5-point moving median
        2. 3-point moving median
        3. Hanning smoothing

    Parameters
    ----------
    data : np.ndarray
        Input 1D numeric sequence.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    np.ndarray
        Smoothed sequence.
    """
    result = moving_median_odd(data, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_95H(data: np.ndarray, endrule: str = "median", **kwargs) -> np.ndarray:
    """
    Tukey's 95H smoothing procedure.

    Steps:
        1. 9-point moving median
        2. 5-point moving median
        3. Hanning smoothing

    Parameters
    ----------
    data : np.ndarray
        Input 1D numeric sequence.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    np.ndarray
        Smoothed sequence.
    """
    result = moving_median_odd(data, 9, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = hanning(result)
    return result


ALGORITHMS: dict[str, Callable] = {
    "4253H": smh_4253H,
    "353QH": smh_353QH,
    "3G53QH": smh_3G53QH,
    "53H": smh_53H,
    "95H": smh_95H,
}


def build_tuckey_fn(algorithm: str, endrule: str = "median") -> Callable:
    """
    Build a Tukey smoothing function from a symbolic algorithm string.

    Examples
    --------
    - "4253H" → predefined 4253H algorithm
    - "353QH" → predefined 353QH algorithm
    - Custom algorithm like "53HQ":
        * "5" → 5-point median
        * "3" → 3-point median
        * "H" → Hanning
        * "Q" → Quadratic interpolation

    Parameters
    ----------
    algorithm : str
        Name of predefined algorithm ("4253H", "353QH", etc.)
        or a custom string of operations.
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".

    Returns
    -------
    Callable
        Function that applies the requested smoothing algorithm.
    """
    def wrap_median(prev_fn, k):
        return lambda d: prev_fn(moving_median_odd(d, k, endrule))

    def wrap_hanning(prev_fn):
        return lambda d: prev_fn(hanning(d))

    def wrap_gaussian_hanning(prev_fn):
        return lambda d: prev_fn(hanning(d, True))

    def wrap_quadratic(prev_fn):
        return lambda d: prev_fn(quadratic_interpolation(d))

    action_wrappers = {"H": wrap_hanning, "G": wrap_gaussian_hanning, "Q": wrap_quadratic}

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


def tukey_smoothing(
    data: np.ndarray,
    algorithm: str,
    endrule: str = "median",
    twice: int = 0,
    var_function: Callable = lambda h: h,
) -> np.ndarray:
    """
    Apply Tukey smoothing to a numeric sequence.

    Parameters
    ----------
    data : np.ndarray
        Input numeric sequence.
    algorithm : str
        Algorithm string (predefined or custom).
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".
    twice : int, optional
        Number of twicing iterations. Default is 0.
    var_function : callable, optional
        Variance adjustment function. Default is identity.

    Returns
    -------
    np.ndarray
        Smoothed sequence.
    """
    fn = build_tuckey_fn(algorithm, endrule)
    mod_data = fn(data)

    if twice > 0:
        mod_data = twicing(data, mod_data, fn, times=twice)

    return mod_data


def boost_hist_tukey(
    hnom: hist.Hist,
    algorithm: str,
    endrule: str = "median",
    twice: int = 0,
    var_function: Callable = lambda h: h,
) -> hist.Hist:
    """
    Apply Tukey smoothing to a boost-histogram.

    Parameters
    ----------
    hnom : hist.Hist
        Input histogram.
    algorithm : str
        Algorithm string (predefined or custom).
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".
    twice : int, optional
        Number of twicing iterations. Default is 0.
    var_function : callable, optional
        Variance adjustment function. Default is identity.

    Returns
    -------
    hist.Hist
        Smoothed histogram.
    """
    data, data_var = hist_extract(hnom)
    mod_data = tukey_smoothing(data=data, algorithm=algorithm, endrule=endrule, twice=twice, var_function=var_function)
    hnom[...] = [(mod_data[i], var_function(data_var[i])) for i in range(len(mod_data))]
    return hnom


def smooth_rebin_monotonic(hnom_hist: hist.Hist, hsys_hist: hist.Hist) -> hist.Hist:
    """
    Smooth histogram ratio with monotonic binning rule.

    Parameters
    ----------
    hnom_hist : hist.Hist
        Nominal histogram.
    hsys_hist : hist.Hist
        Systematic histogram.

    Returns
    -------
    hist.Hist
        Smoothed histogram with monotonic binning.
    """
    return boost_hist_rebin(hnom_hist, hsys_hist, nmax=0)


def smooth_rebin_parabolic(hnom_hist: hist.Hist, hsys_hist: hist.Hist) -> hist.Hist:
    """
    Smooth histogram ratio with parabolic binning rule.

    Parameters
    ----------
    hnom_hist : hist.Hist
        Nominal histogram.
    hsys_hist : hist.Hist
        Systematic histogram.

    Returns
    -------
    hist.Hist
        Smoothed histogram with parabolic binning.
    """
    return boost_hist_rebin(hnom_hist, hsys_hist, nmax=1)


def smooth_hist_general(
    hnom_hist: hist.Hist,
    algorithm: str,
    hsys_hist: hist.Hist | None = None,
    endrule: str = "median",
    twice: int = 0,
) -> hist.Hist:
    """
    Apply a general histogram smoothing procedure.

    Parameters
    ----------
    hnom_hist : hist.Hist
        Input nominal histogram.
    algorithm : str
        Smoothing algorithm ("monotonic", "parabolic", or Tukey string).
    hsys_hist : hist.Hist, optional
        Systematic histogram (used for rebinning).
    endrule : {"median", "keep"}, optional
        Rule for handling boundaries in median smoothing.
        Default is "median".
    twice : int, optional
        Number of twicing iterations. Default is 0.

    Returns
    -------
    hist.Hist
        Smoothed histogram.
    """
    hnom_hist = hnom_hist.copy()

    if algorithm == "monotonic":
        hnew = smooth_rebin_monotonic(hnom_hist, hsys_hist)
    elif algorithm == "parabolic":
        hnew = smooth_rebin_parabolic(hnom_hist, hsys_hist)
    else:
        hnew = boost_hist_tukey(hnom_hist, algorithm, endrule=endrule, twice=twice)
    return hnew


def hist_rebin(
    hnom: np.ndarray,
    hsys: np.ndarray,
    hnom_err: np.ndarray,
    nmax: int,
    apply_smooth: bool = True,
) -> np.ndarray:
    """
    Rebin histograms using ratio smoothing.

    Parameters
    ----------
    hnom : np.ndarray
        Nominal histogram bin contents.
    hsys : np.ndarray
        Systematic histogram bin contents.
    hnom_err : np.ndarray
        Errors on nominal histogram bins.
    nmax : int
        Max number of extrema allowed after merging.
    apply_smooth : bool, optional
        Whether to smooth the ratio histogram. Default is True.

    Returns
    -------
    np.ndarray
        Smoothed nominal histogram.
    """
    bins = get_local_extrema_binning(hnom, hsys, hnom_err, nmax)
    ratio = get_ratio_hist(hnom, hsys, bins)

    if apply_smooth and len(ratio) > 2:
        ratio = apply_smoothing_kernel(ratio)

    smoothed = ratio * hnom

    norm_old, norm_new = np.sum(hsys), np.sum(smoothed)
    if norm_new > 1e-9:
        smoothed *= norm_old / norm_new

    return smoothed


def boost_hist_rebin(
    hnom_hist: hist.Hist,
    hsys_hist: hist.Hist,
    nmax: int,
    apply_smooth: bool = True,
) -> hist.Hist:
    """
    Rebin two histograms using ratio smoothing.

    Parameters
    ----------
    hnom_hist : hist.Hist
        Nominal histogram.
    hsys_hist : hist.Hist
        Systematic histogram.
    nmax : int
        Max number of extrema allowed after merging.
    apply_smooth : bool, optional
        Whether to smooth the ratio histogram. Default is True.

    Returns
    -------
    hist.Hist
        Smoothed nominal histogram.
    """
    hnom, hnom_err = hist_extract(hnom_hist)
    hsys, _ = hist_extract(hsys_hist)
    hnom_err = np.sqrt(hnom_err)

    smoothed = hist_rebin(hnom=hnom, hsys=hsys, hnom_err=hnom_err, nmax=nmax, apply_smooth=apply_smooth)

    hnom_hist[...] = [(i, 0) for i in smoothed]
    return hnom_hist


def hist_extract(h: hist.Hist) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract values and variances from a histogram.

    Parameters
    ----------
    h : hist.Hist
        Input histogram.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - Values
        - Variances (zero if not available)
    """
    v = h.view()
    if hasattr(v, "variance"):
        return np.asarray(v.value), np.asarray(v.variance)
    else:
        return np.asarray(v), np.zeros_like(v)
