try:
    import ROOT
except ModuleNotFoundError:
    raise SystemExit("Can't find ROOT. Please install ROOT.")

from hist_smooth.smoothing_api import tukey_smoothing, hist_rebin

def root_smooth_hist_general(hnom_hist, algorithm, hsys_hist=None,
                         apply_smooth=True, endrule='median',
                         twice=0):
    
    hnom_hist = hnom_hist.Clone(hnom_hist.GetName() + "_smooth")
    if algorithm == "monotonic":
        hnew = root_rebin(hnom_hist, hsys_hist, nmax=0)
    elif algorithm == "parabolic":
	    hnew = root_rebin(hnom_hist, hsys_hist, nmax=1)
    else:
        hnew = root_hist_tukey(hnom_hist, algorithm, endrule=endrule, twice=twice)
    return hnew

def root_hist_tukey(hnom, algorithm, endrule='median', twice=0, var_function=lambda h: h):
    nbins = hnom.GetNbinsX()
    data = [hnom.GetBinContent(i) for i in range(1, nbins + 1)]
    data_var = [hnom.GetBinError(i)**2 for i in range(1, nbins + 1)]

    mod_data = tukey_smoothing(data=data, algorithm=algorithm, endrule=endrule, twice=twice, var_function=var_function)
    
    for i in range(1, nbins + 1):
        hnom.SetBinContent(i, mod_data[i-1])
        hnom.SetBinError(i, (var_function(data_var)[i-1])**0.5)
    
    return hnom

def root_rebin(hnom_hist, hsys_hist, nmax, apply_smooth=True):
    nbins = hnom_hist.GetNbinsX()
    hnom = [hnom_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    hsys = [hsys_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    hnom_err = [hsys.GetBinError(i) for i in range(1, nbins + 1)]

    smoothed = hist_rebin(hnom=hnom, hsys=hsys, hnom_err=hnom_err, nmax=nmax, apply_smooth=apply_smooth)

    for i in range(1, nbins + 1):
        hnom.SetBinContent(i, smoothed[i-1])
        hnom.SetBinError(i, 0)
    return hnom