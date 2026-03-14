# hist_smooth

Library for smoothing histograms with a focus on reproducible behavior and simple integration into analysis pipelines.

## Installation

From source:

```bash
pip install -e .
```

## Quickstart

```python
import hist
import boost_histogram as bh
import hist_smooth

def make_hist(values, variances):
    n = len(values)
    h = hist.Hist(hist.axis.Regular(n, 0, n), storage=bh.storage.Weight())
    h[...] = list(zip(values, variances))
    return h

hnom = make_hist([10.0, 12.0, 11.0], [10.0, 12.0, 11.0])
hsys = make_hist([11.0, 9.0, 13.0], [11.0, 9.0, 13.0])

hmod = hist_smooth.smooth_hist_general(hnom, "parabolic", hsys_hist=hsys)
```

## Algorithms

Supported algorithms include:

- monotonic
- parabolic
- 3
- 353HQ

Check the source for the full list and implementation details.

## Testing

```bash
pytest
```
