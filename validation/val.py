import hist_smooth

import time
import numpy as np
import hist
import boost_histogram as bh
import hist_smooth

def make_hist(n_bins, lam=100):
    """Create a boost-histogram filled with Poisson data."""
    data = np.random.poisson(lam=lam, size=n_bins)
    hst = hist.Hist(
        hist.axis.Regular(n_bins, -10, 10),
        storage=bh.storage.Weight()
    )
    hst[...] = [(val, val) for val in data]
    return hst

def make_hist2(n_bins, lam=100):
    data = np.random.poisson(lam=lam, size=n_bins)
    hst = hist.Hist(
        hist.axis.Regular(n_bins, -10, 10),
    ).fill(data)
    return hst

def benchmark_algorithm(algorithm, n_bins, n_repeats=3, **kwargs):
    hnom_hist = make_hist(n_bins)

    if algorithm in ("monotonic", "parabolic"):
        hsys_hist = make_hist(n_bins, lam=105)
        kwargs = {**kwargs, "hsys_hist": hsys_hist}
    
    hist_smooth.smooth_hist_general(hnom_hist, algorithm, **kwargs)
    
    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        hist_smooth.smooth_hist_general(hnom_hist, algorithm, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times)

def find_bin_count_for_one_second(algorithm, bin_counts=None, **kwargs):
    if bin_counts is None:
        bin_counts = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]

    results = []
    for n in bin_counts:
        t = benchmark_algorithm(algorithm, n, **kwargs)
        results.append((n, t))
        print(f"{algorithm} - {n} bins: {t:.4f} s")

    bins, times = zip(*results)
    coeffs = np.polyfit(bins, times, 1)  
    slope, intercept = coeffs
    
    target_bins = int((1.0 - intercept) / slope) if slope > 0 else None
    print(f"\nEstimated ~{target_bins} bins for {algorithm} to take ~1s\n")
    
    return results, target_bins

for algo in ["3", "353HQ"]:
    bin_counts = [1167540, 162274]  

    results, target_bins = find_bin_count_for_one_second(algo, bin_counts=bin_counts)

