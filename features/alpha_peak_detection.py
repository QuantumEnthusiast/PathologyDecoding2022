from scipy.optimize import curve_fit
import numpy as np
from background_features import WelchEstimate
import config as c

def OptimizePdrCurve(EEG, ch):
    def func_bg(f, B, C):
        return B - C * np.log10(f)

    def func_pk1(f, A1, F1, D1):
        Ppk1 = A1 * np.exp((-(f - F1) ** 2) / (D1 ** 2))
        Pbg = B - C * np.log10(f)
        return Ppk1 + Pbg

    def func_pk2(f, A2, F2, D2):
        Ppk2 = A2 * np.exp((-(f - F2) ** 2) / (D2 ** 2))
        Ppk1 = A1 * np.exp((-(f - F1) ** 2) / (D1 ** 2))
        Pbg = B - C * np.log10(f)
        return Ppk2 + Ppk1 + Pbg

    fmin = 8
    fmax = 12

    Plog, f_bound = WelchEstimate(EEG[ch], c.fs, c.NOVERLAP, c.NFFT, c.WINDOW, fmin, fmax)

    # {B,C} argmin |Plog - Pbg|
    popt, _ = curve_fit(func_bg, f_bound, Plog)
    B = popt[0]
    C = popt[1]

    # {B,A1,F1,D1,C} argmin |Pk1 - Plog - Pbg|
    popt, _ = curve_fit(func_pk1, f_bound, Plog)
    A1 = popt[0]
    F1 = popt[1]
    D1 = popt[2]

    # {B,A1,A2,F1,F2,D1,D2,C} argmin |Pk2 - Pk1 - Plog - Pbg|
    popt, _ = curve_fit(func_pk2, f_bound, Plog)
    A2 = popt[0]
    F2 = popt[1]
    D2 = popt[2]

    print('B = ', B)
    print('C = ', C)
    print('A1 = ', A1)
    print('F1 = ', F1)
    print('D1 = ', D1)
    print('A2 = ', A2)
    print('F2 = ', F2)
    print('D2 = ', D2)

    return B, C, A1, F1, D1, A2, F2, D2