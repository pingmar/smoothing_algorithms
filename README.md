# hist-smooth

A Python package providing utilities for **smoothing histograms**.  
It supports both **Boost-Histogram** and  **ROOT** histograms.

---

## Installation

You can install directly from GitHub:

```bash
pip install git+https://github.com/pingmar/hist_smooth.git
```

Ensure you have Python ≥3.10.
Optional dependency: Install ROOT if you want to use the functions in this library that work with ROOT histograms.

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

## Example Usage of smooth_hist_general

```python
import hist_smooth
import numpy as np
import hist
import boost_histogram as bh

np.random.seed(42)

data = np.random.poisson(lam=100, size=50)

hst = hist.Hist(
    hist.axis.Regular(50, -10, 10),
    storage=bh.storage.Weight()
)

hst[...] = [(val, val*2) for val in data]

hst_smth = hist_smooth.smooth_hist_general(hst, '353')

hst.plot()
hst_smth.plot()
```

## Example Usage of plot_grid_with_smoothing

```python
import numpy as np
import hist_smooth

# Generate data
data = np.random.normal(loc=0, scale=1, size=10_000)

# Define binning
binnings = [(50, -5, 5), (30, -3, 3)]

# Define algorithms
algorithms = ["353HQ", "monotonic", "parabolic"]

# Plot grid
hist_smooth.plot_grid_with_smoothing(
    data_list=[data],
    binnings=binnings,
    algorithms=algorithms,
    endrule="median",
    twice=1
)
```

This will create a **comparison grid** of smoothed histograms, including ratio panels.

## Example Usage of root_smooth_hist_general

```python 
import ROOT
import numpy as np
from hist_smooth.root_smoothing import root_smooth_hist_general

np.random.seed(42)

data = np.random.poisson(lam=100, size=50)

hst = ROOT.TH1F("hst", "Example Histogram", 50, -10, 10)

for i in range(1, 50 + 1):
    hst.SetBinContent(i, data[i - 1])
    hst.SetBinError(i, (data[i - 1]) ** 0.5)

hst_smth = root_smooth_hist_general(hst, "353")

#visualization

hst.SetLineColor(ROOT.kBlack)
hst.SetLineWidth(2)
hst.SetMarkerStyle(20)
hst.SetMarkerSize(0.8)

hst_smth.SetLineColor(ROOT.kRed)
hst_smth.SetLineWidth(2)
hst_smth.SetMarkerStyle(24)
hst_smth.SetMarkerSize(0.8)

c = ROOT.TCanvas("c", "Histograms", 800, 600)
hst.Draw("HIST SAME")
hst_smth.Draw("HIST SAME")

leg = ROOT.TLegend(0.65, 0.15, 0.88, 0.30)
leg.AddEntry(hst, "Original", "lep")
leg.AddEntry(hst_smth, "Smoothed", "l")
leg.Draw()

c.SaveAs("histograms.png")
```

---

## Features

* Tukey smoothing methods
* Rebinning (monotonic/parabolic) smoothing methods
* Works with **boost-histogram** and **hist**
* ROOT histogram support
* Statistical validation (reduced chi², KS test)
* Visualization tools for quick comparison

---

## License

MIT License.
See [LICENSE](./LICENSE) for details.
