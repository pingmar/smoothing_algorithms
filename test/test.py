import pytest
import numpy as np
import hist
import boost_histogram as bh
import hist_smooth

def make_test_hist(values, variances):
    n_bins = len(values)
    hst = hist.Hist(hist.axis.Regular(n_bins, 0, n_bins), storage=bh.storage.Weight())
    hst[...] = list(zip(values, variances))
    return hst

@pytest.mark.parametrize("algorithm,expected_vals,expected_vars", [
    ("monotonic",
     [119.16667175292969, 119.16667175292969, 119.16667175292969, 119.16667175292969,
      119.16667175292969, 118.75, 117.91666412353516, 117.49999237060547,
      117.49999237060547, 117.49999237060547],
     [0.0]*10),
    ("parabolic",
     [92.5, 102.5, 122.5, 132.5, 132.5, 128.75, 121.25, 117.49999237060547,
      117.49999237060547, 117.49999237060547],
     [0.0]*10)
])

def test_smooth_hist_general(algorithm, expected_vals, expected_vars):
    hnom_vals = [100.0]*10
    hnom_vars = [100.0]*10
    hsys_vals = [105.0, 80.0, 120.0, 160.0, 140.0, 110.0, 180.0, 100.0, 100.0, 90.0]
    hsys_vars = [105.0, 80.0, 120.0, 160.0, 140.0, 110.0, 180.0, 100.0, 100.0, 90.0]

    hnom_hist = make_test_hist(hnom_vals, hnom_vars)
    hsys_hist = make_test_hist(hsys_vals, hsys_vars)

    hmod = hist_smooth.smooth_hist_general(hnom_hist, algorithm, hsys_hist=hsys_hist)

    vals = [x.value for x in hmod.view(flow=False)]
    vars_ = [x.variance for x in hmod.view(flow=False)]

    np.testing.assert_allclose(vals, expected_vals)
    np.testing.assert_allclose(vars_, expected_vars)
    print(vals)
    