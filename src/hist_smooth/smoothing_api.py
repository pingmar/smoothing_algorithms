from .base_smoothing import (
    moving_median_odd,
    hanning,
    quadratic_interpolation,
    twicing,
    even_median_four,
    even_median_two,
)

from .tukey_smoothing import (
    ALGORITHMS
)

import numpy as np

def Smoothing(data, algorithm, endrule='median', twice=0):

    algorithms = ALGORITHMS

    if algorithm not in algorithms:
        fn = lambda d: d 
        for action in algorithm[::-1]:
            if action.isdigit():
                fn = (lambda prev_fn, k_val, e_rule:
                      lambda d: prev_fn(moving_median_odd(d, k_val, e_rule))
                     )(fn, int(action), endrule)
            elif action.isalpha():
                if action == 'H':
                    fn = (lambda prev_fn:
                          lambda d: prev_fn(hanning(d))
                         )(fn)
                elif action == 'G':
                    fn = (lambda prev_fn:
                          lambda d: prev_fn(hanning(d, True))
                         )(fn)
                elif action == 'Q':
                    fn = (lambda prev_fn:
                          lambda d: prev_fn(quadratic_interpolation(d))
                         )(fn)
                else:
                    raise ValueError(f"Invalid action '{action}'")
    else:
        fn = algorithms[algorithm]

    mod_data = fn(data)
    if twice > 0:
        mod_data = twicing(data, mod_data, fn, times=twice)
    return mod_data
