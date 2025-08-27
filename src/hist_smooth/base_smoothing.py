import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def moving_median_odd(data, window, endrule='median'):
    """Moving median with odd window size."""
    data = np.asarray(data, dtype=float)
    n = len(data)
    k2 = window // 2
    new_data = np.zeros(n, dtype=float)
    core = np.median(sliding_window_view(data, window), axis=1)
    new_data[k2:n - k2] = core

    if endrule == 'keep':
        new_data[:k2] = data[:k2]
        new_data[-k2:] = data[-k2:]
    elif endrule == 'median':
        for i in range(k2):
            new_data[i] = np.median(data[:2 * i + 1])
            new_data[-i - 1] = np.median(data[-(2 * i + 1):])
    else:
        raise ValueError(f"Unknown endrule: {endrule}")

    return new_data


def hanning(data, condition=False):
    """Hanning smoother (0.25, 0.5, 0.25 weights)."""
    data = np.asarray(data, dtype=float)
    windows = sliding_window_view(data, window_shape=3)
    smoothed_core = []

    for triplet in windows:
        a, b, c = triplet
        if condition:
            signs = np.sign([a, b, c])
            if np.all(signs != 0) and signs[0] == -signs[1] and signs[1] == -signs[2]:
                smoothed_core.append(0.25 * a + 0.5 * b + 0.25 * c)
            else:
                smoothed_core.append(b)
        else:
            smoothed_core.append(0.25 * a + 0.5 * b + 0.25 * c)

    smoothed_core = np.array(smoothed_core)
    result = np.empty_like(data)
    result[1:-1] = smoothed_core
    result[0] = data[0]
    result[-1] = data[-1]

    return result


def quadratic_interpolation(y):
    """Quadratic interpolation for flat regions."""
    y_out = np.array(y, float).copy()
    for i in range(2, len(y)-2):
        if (y[i-1] == y[i] == y[i+1]) and ((y[i-2] < y[i] and y[i+2] < y[i]) or (y[i-2] > y[i] and y[i+2] > y[i])):
            left_dist = abs(y[i-2] - y[i])
            right_dist = abs(y[i+2] - y[i])
            if left_dist >= right_dist:
                x_fit = [i - 2, i - 1, i + 2]
                y_fit = [y[i-2], y[i-1], y[i+2]]
                indices_to_update = [i, i + 1]
            else:
                x_fit = [i - 2, i + 1, i + 2]
                y_fit = [y[i-2], y[i+1], y[i+2]]
                indices_to_update = [i - 1, i]
            coeffs = np.polyfit(x_fit, y_fit, 2)
            poly_func = np.poly1d(coeffs)
            for index in indices_to_update:
                y_out[index] = poly_func(index)
    return np.array(y_out, float)


def twicing(original_data, smooth, algorithm, times=1):
    """Apply twicing procedure."""
    for _ in range(times):
        rough = original_data - smooth
        smooth = smooth + algorithm(rough)
    return smooth


def even_median_four(data):
    """Even median with window size 4."""
    new_data = np.zeros(len(data) + 1, dtype=float)
    core = np.median(sliding_window_view(data, 4), axis=1)
    new_data[0] = data[0]
    new_data[1] = (data[1] + data[0])/2
    new_data[2:-2] = core
    new_data[-2] = (data[-2] + data[-1])/2
    new_data[-1] = data[-1]
    return new_data


def even_median_two(data):
    """Even median with window size 2."""
    return np.median(sliding_window_view(data, 2), axis=1)
