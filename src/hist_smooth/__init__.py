from hist_smooth.grid_plot_hist import plot_grid_with_smoothing
from hist_smooth.smoothing_api import smooth_hist_general

try:
    from hist_smooth.root_smoothing import root_smooth_hist_general 
except ModuleNotFoundError:
    print("ROOT smoothing is not available. Please install ROOT")
    root_smooth_hist_general = None