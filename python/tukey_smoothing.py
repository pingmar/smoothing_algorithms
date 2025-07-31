from .base_smoothing import (
    moving_median_odd, hanning, even_median_four, 
    even_median_two, quadratic_interpolation
)


def smh_4253H(data, endrule='median', **kwargs):
    result = even_median_four(data)
    result = even_median_two(result)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_353QH(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 3, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_3G53QH(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 3, endrule)
    result = hanning(result, condition=True)
    result = moving_median_odd(result, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = quadratic_interpolation(result)
    result = hanning(result)
    return result


def smh_53H(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 5, endrule)
    result = moving_median_odd(result, 3, endrule)
    result = hanning(result)
    return result


def smh_95H(data, endrule='median', **kwargs):
    result = moving_median_odd(data, 9, endrule)
    result = moving_median_odd(result, 5, endrule)
    result = hanning(result)
    return result

ALGORITHMS = {
    '4253H': smh_4253H,
    '353QH': smh_353QH,
    '3G53QH': smh_3G53QH,
    '53H': smh_53H,
    '95H': smh_95H,
}
