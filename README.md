# hist-smooth

A Python package providing utilities for **smoothing histograms**.  
It supports both **Boost-Histogram** and  **ROOT** histograms.

---

## Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/pingmar/hist_smooth.git
```

Ensure you have Python ≥3.10, optional `ROOT` for ROOT histogram support

---

## Main Functions

### 1. `smooth_hist_general`

```python
from hist_smooth.smoothing_api import smooth_hist_general
```

General-purpose smoothing function for **boost-histogram** or **hist** objects.

---
### 2. `plot_grid_with_smoothing`

```python
from hist_smooth.grid_plot_hist import plot_grid_with_smoothing
```

Grid plotting utility for comparing multiple **datasets, binnings, and algorithms**.
Displays a grid of plots with smoothed histograms and ratio panels.

---

## Example Usage

```python
import numpy as np
import hist
from hist_smooth.smoothing_api import smooth_hist_general
from hist_smooth.grid_plot_hist import plot_grid_with_smoothing

# Generate data
data = np.random.normal(loc=0, scale=1, size=10_000)

# Define binning
binnings = [(50, -5, 5), (30, -3, 3)]

# Define algorithms
algorithms = ["353HQ", "monotonic", "parabolic"]

# Plot grid
plot_grid_with_smoothing(
    data_list=[data],
    binnings=binnings,
    algorithms=algorithms,
    endrule="median",
    twice=1
)
```

This will create a **comparison grid** of smoothed histograms, including ratio panels.

---

## Features

* Tukey smoothing methods
* ARebinning (monotonic/parabolic) smoothing methods
* Works with **boost-histogram** and **hist**
* ROOT histogram support
* Statistical validation (reduced chi², KS test)
* Visualization tools for quick comparison

---

## License

MIT License.
See [LICENSE](./LICENSE) for details.

```
