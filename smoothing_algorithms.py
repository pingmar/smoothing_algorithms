import numpy as np
import skhep_testdata, uproot

def three_median(y):
    z = []
    for i in range(1, len(y)-1):
        z_i = np.median((y[i-1], y[i], y[i+1]))
        z.append(z_i)
    z_1 = np.median((3*z[1] - 2*z[2], y[0], z[1]))
    z_n = np.median((z[-1], y[-1], 3*z[-1] - 2*z[-2]))
    z = [z_1] + z + [z_n]
    return np.array(z, float)

def five_median(y):
    z = []
    for i in range(2, len(y)-2):
        z_i = np.median((y[i-2], y[i-1], y[i], y[i+1], y[i+2]))
        z.append(z_i)
    z_1 = y[0]
    z_n = y[-1]
    z_2 = np.median((y[0], y[1], y[2]))
    z_nm1 = np.median((y[-3], y[-2], y[-1]))
    z = [z_1] + [z_2] + z + [z_nm1] +  [z_n]
    return np.array(z, float)

def quadratic_interpolation(y):
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

def hanning(data, condition=False):
    data = np.asarray(data, dtype=float)
    smoothed = []

    for i in range(1, len(data) - 1):
        a, b, c = data[i - 1], data[i], data[i + 1]
        if condition:
            signs = np.sign([a, b, c])
            if signs[0] != 0 and signs[1] != 0 and signs[2] != 0:
                if signs[0] == -signs[1] and signs[1] == -signs[2]:
                    smoothed.append(0.25 * a + 0.5 * b + 0.25 * c)
                else:
                    smoothed.append(b)
            else:
                smoothed.append(b)
        else:
            smoothed.append(0.25 * a + 0.5 * b + 0.25 * c)

    smoothed = [data[0]] + smoothed + [data[-1]]
    return np.array(smoothed, float)

def smth_353QH(data):
    return three_median(five_median(three_median(quadratic_interpolation(hanning(data)))))

def smth_3G53QH(data):
    return three_median(hanning(five_median(three_median(quadratic_interpolation(hanning(data)))), True))

def twicing(original_data, smooth, algorithm, times=1):
    for _ in range(times):
        rough = original_data - smooth
        smooth = smooth + algorithm(rough)
    return smooth

def print_both(data, mod_data):
    h = hist.Hist(hist.axis.Regular(120, 60, 120))

    h.fill(data)

    h.plot()

    h2 = hist.Hist(hist.axis.Regular(120, 60, 120))
    h2.fill(mod_data)

    h2.plot()
    return 0




#Use example
tree = uproot.open(skhep_testdata.data_path("uproot-Zmumu.root"))["events"]
data = tree["M"].array()

md = twicing(data, smth_353QH(data), smth_353QH, times=8)
print_both(data, md)
