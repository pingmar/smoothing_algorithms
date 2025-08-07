import matplotlib.pyplot as plt
import numpy as np
import hist
from .smoothing_api import Smoothing
from .statistics_utils import reduced_chi2, ks_2samp

def plot_grid_with_smoothing(data_list, binnings, algorithms, figsize=(5, 4)):

    N = len(data_list)
    M = len(binnings)

    fig, axs = plt.subplots(2 * N, M, figsize=(figsize[0] * M, figsize[1] * N * 2),
                            gridspec_kw={'height_ratios': [3, 1] * N})

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
            h = hist.Hist(hist.axis.Regular(bins_number, s, e)).fill(data)
            bin_counts = h.counts()
            variances = h.variances()

            ax_main = axs[i, 0, j]
            ax_ratio = axs[i, 1, j]

            h.plot(ax=ax_main, label='Original')

            for algo_fn in algorithms:
                mod_data = Smoothing(bin_counts, algo_fn, twice=1)

                ratio_val = np.sum(mod_data) / np.sum(bin_counts) if np.sum(bin_counts) != 0 else np.nan
                ks_stat, ks_pvalue = ks_2samp(bin_counts, mod_data)
                chi2_val = reduced_chi2(bin_counts, mod_data, variances)

                h2 = hist.Hist(hist.axis.Regular(bins_number, s, e))
                h2[...] = mod_data

                main_plot_artists = h2.plot(ax=ax_main, label=f"{algo_fn} - ({ratio_val:.2f}, {ks_pvalue:.3f}, {chi2_val:.2f})")
                line_color = main_plot_artists[0][0].get_edgecolor()

                x = h.axes[0].centers

                ratio = np.divide(mod_data, bin_counts,
                                  out=np.zeros_like(mod_data, dtype=float),
                                  where=bin_counts!=0)
                ax_ratio.plot(x, ratio, 'o-', label=algo_fn, color=line_color)

            ax_main.set_title(f"Dist {i+1}, Bin {bins_number}")
            ax_main.legend(fontsize=8)
            ax_ratio.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
            ax_ratio.set_ylim(0, 2)
            ax_ratio.set_ylabel("Ratio")

    plt.tight_layout()
    plt.show()
