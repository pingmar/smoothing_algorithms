import ROOT
from typing import Callable, Optional
from hist_smooth.smoothing_api import tukey_smoothing, hist_rebin


def root_smooth_hist_general(
    hnom_hist: ROOT.TH1,
    algorithm: str,
    hsys_hist: Optional[ROOT.TH1] = None,
    endrule: str = "median",
    twice: int = 0
) -> ROOT.TH1:
    """
    Apply a smoothing or rebinning algorithm to a ROOT histogram.

    Parameters
    ----------
    hnom_hist : ROOT.TH1
        The nominal histogram to be smoothed.
    algorithm : str
        The smoothing algorithm to use.
        Options:
        - "monotonic": merge bins until a monotonic ratio is achieved.
        - "parabolic": merge bins until a parabolic ratio is achieved.
        - any other string: use Tukey-style smoothing with the given name.
    hsys_hist : ROOT.TH1, optional
        Systematic histogram to guide rebinning, required for monotonic/parabolic modes.
        Default is None.
    endrule : str, optional
        End rule for Tukey smoothing (e.g., "median", "mean").
        Default is "median".
    twice : int, optional
        Number of times to apply the smoothing procedure (repeated smoothing).
        Default is 0.

    Returns
    -------
    ROOT.TH1
        A cloned and smoothed ROOT histogram.
    """
    hnom_hist = hnom_hist.Clone(hnom_hist.GetName() + "_smooth")
    if algorithm == "monotonic":
        hnew = root_rebin(hnom_hist, hsys_hist, nmax=0)
    elif algorithm == "parabolic":
        hnew = root_rebin(hnom_hist, hsys_hist, nmax=1)
    else:
        hnew = root_hist_tukey(hnom_hist, algorithm, endrule=endrule, twice=twice)
    return hnew


def root_hist_tukey(
    hnom: ROOT.TH1,
    algorithm: str,
    endrule: str = "median",
    twice: int = 0,
    var_function: Callable[[list[float]], list[float]] = lambda h: h
) -> ROOT.TH1:
    """
    Apply Tukey-style smoothing to a ROOT histogram.

    Parameters
    ----------
    hnom : ROOT.TH1
        Input histogram to be smoothed (modified in place).
    algorithm : str
        Tukey smoothing algorithm to apply ("tukey", "running_mean", etc.).
    endrule : str, optional
        End rule for smoothing ("median", "mean"), by default "median".
    twice : int, optional
        Number of additional smoothing passes to apply, by default 0.
    var_function : callable, optional
        Function applied to bin variances before smoothing.
        Must accept a list of variances and return a list of transformed variances.
        By default, identity function.

    Returns
    -------
    ROOT.TH1
        Histogram with smoothed bin contents and updated errors.
    """
    nbins = hnom.GetNbinsX()
    data = [hnom.GetBinContent(i) for i in range(1, nbins + 1)]
    data_var = [hnom.GetBinError(i)**2 for i in range(1, nbins + 1)]

    mod_data = tukey_smoothing(
        data=data, algorithm=algorithm, endrule=endrule,
        twice=twice, var_function=var_function
    )
    mod_var = var_function(data_var)

    for i in range(1, nbins + 1):
        hnom.SetBinContent(i, mod_data[i - 1])
        hnom.SetBinError(i, (mod_var[i - 1]) ** 0.5)

    return hnom


def root_rebin(
    hnom_hist: ROOT.TH1,
    hsys_hist: Optional[ROOT.TH1],
    nmax: int,
    apply_smooth: bool = True
) -> ROOT.TH1:
    """
    Rebin a ROOT histogram using systematic guidance and statistical criteria.

    Parameters
    ----------
    hnom_hist : ROOT.TH1
        Nominal histogram to rebin.
    hsys_hist : ROOT.TH1, optional
        Systematic histogram used for ratio/extrema calculation.
        Must be provided for rebinning. Default is None.
    nmax : int
        Maximum allowed number of extrema in the ratio histogram.
        - 0 for monotonic smoothing
        - 1 for parabolic smoothing
    apply_smooth : bool, optional
        If True, apply smoothing after rebinning. Default is True.

    Returns
    -------
    ROOT.TH1
        Histogram with modified bin contents after rebinning.
    """
    nbins = hnom_hist.GetNbinsX()
    hnom = [hnom_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    hsys = [hsys_hist.GetBinContent(i) for i in range(1, nbins + 1)] if hsys_hist else [0] * nbins
    hnom_err = [hnom_hist.GetBinError(i) for i in range(1, nbins + 1)]

    smoothed = hist_rebin(
        hnom=hnom, hsys=hsys, hnom_err=hnom_err,
        nmax=nmax, apply_smooth=apply_smooth
    )

    for i in range(1, nbins + 1):
        hnom_hist.SetBinContent(i, smoothed[i - 1])
        hnom_hist.SetBinError(i, 0.0)

    return hnom_hist
