import matplotlib.pyplot as plt
import numpy as np
import hist
from hist_smooth.smoothing_api import smooth_hist_general
from hist_smooth.statistics_utils import reduced_chi2, ks_2samp
import boost_histogram as bh
from collections.abc import Callable
from typing import Sequence

def plot_grid_with_smoothing(
    data_list: Sequence[np.ndarray],
    binnings: Sequence[tuple[int, float, float]],
    algorithms: Sequence[Callable],
    figsize: tuple[float, float] = (5, 4),
    hsys_hist: hist.Hist | None = None,
    apply_smooth: bool = True,
    endrule: str = "median",
    twice: int = 0
) -> None:
    """
    Plot histograms in a grid layout with different smoothing algorithms applied.

    Parameters
    ----------
    data_list : sequence of array_like
        A list of datasets (1D arrays) to be histogrammed and smoothed.
        Each dataset will correspond to a row in the plot grid.
    binnings : sequence of tuple of (int, float, float)
        Histogram binning configuration for each column.
        Each tuple is (number_of_bins, start, end).
    algorithms : sequence of callable
        A list of smoothing algorithm functions.
        Each algorithm must accept a histogram (`hist.Hist`) and return a smoothed version.
    figsize : tuple of float, optional
        Base size of each subplot (width, height).
        Default is (5, 4).
    hsys_hist : hist.Hist or None, optional
        Optional systematic histogram used in smoothing.
        Default is None.
    apply_smooth : bool, optional
        Whether to actually apply smoothing (True) or just plot original histograms.
        Default is True.
    endrule : str, optional
        Boundary handling rule passed to the smoothing algorithm.
        Options typically include {"median", "keep"}.
        Default is "median".
    twice : int, optional
        Number of twicing (iterative smoothing refinement) iterations.
        Default is 0.

    Returns
    -------
    None
        Displays a matplotlib grid of histograms and ratio plots.

    Notes
    -----
    - The figure has shape `(2 * N, M)` where:
        * `N = len(data_list)` (rows of datasets),
        * `M = len(binnings)` (columns for binning configs).
    - For each dataset and binning, the plot shows:
        * Top panel: histogram + smoothed versions
        * Bottom panel: ratio of smoothed/original bin contents
    - Statistics shown in legend for each algorithm:
        * Normalization ratio,
        * KS-test p-value,
        * Reduced χ² value.
    """
    N = len(data_list)
    M = len(binnings)

    fig, axs = plt.subplots(
        2 * N, M,
        figsize=(figsize[0] * M, figsize[1] * N * 2),
        gridspec_kw={"height_ratios": [3, 1] * N}
    )

    if N == 1 and M == 1:
        axs = np.array([[[axs[0]], [axs[1]]]])
    elif N == 1:
        axs = np.array([[axs[0, :], axs[1, :]]])
    elif M == 1:
        axs_reshaped = np.empty((N, 2, 1), dtype=object)
        for i in range(N):
            axs_reshaped[i, 0, 0] = axs[2 * i]
            axs_reshaped[i, 1, 0] = axs[2 * i + 1]
        axs = axs_reshaped
    else:
        axs = axs.reshape(N, 2, M)

    for i, data in enumerate(data_list):
        for j, bin_cfg in enumerate(binnings):
            bins_number, s, e = bin_cfg
            h = hist.Hist(hist.axis.Regular(bins_number, s, e),
                          storage=bh.storage.Weight()).fill(data)
            bin_counts = h.counts()
            variances = h.variances()

            ax_main = axs[i, 0, j]
            ax_ratio = axs[i, 1, j]

            h.plot(ax=ax_main, label="Original")

            for algo_fn in algorithms:
                #h2 = hist.Hist(hist.axis.Regular(bins_number, s, e),
                #               storage=bh.storage.Weight())
                hsys_hist = h # need to fix
                h2 = smooth_hist_general(
                    h,
                    algorithm=algo_fn,
                    hsys_hist=hsys_hist,
                    endrule=endrule,
                    twice=twice,
                )
                mod_data = h2.values()

                ratio_val = np.sum(mod_data) / np.sum(bin_counts) if np.sum(bin_counts) != 0 else np.nan
                ks_stat, ks_pvalue = ks_2samp(bin_counts, mod_data)
                chi2_val = reduced_chi2(bin_counts, mod_data, variances)

                main_plot_artists = h2.plot(ax=ax_main,
                                            label=f"{algo_fn} - ({ratio_val:.2f}, {ks_pvalue:.3f}, {chi2_val:.2f})")
                line_color = main_plot_artists[0][0].get_edgecolor()

                x = h.axes[0].centers
                ratio = np.divide(
                    mod_data, bin_counts,
                    out=np.zeros_like(mod_data, dtype=float),
                    where=bin_counts != 0
                )
                ax_ratio.plot(x, ratio, "o-", label=algo_fn, color=line_color)

            ax_main.set_title(f"Dist {i+1}, Bin {bins_number}")
            ax_main.legend(fontsize=8)
            ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
            ax_ratio.set_ylim(0, 2)
            ax_ratio.set_ylabel("Ratio")

    plt.tight_layout()
    plt.show()
