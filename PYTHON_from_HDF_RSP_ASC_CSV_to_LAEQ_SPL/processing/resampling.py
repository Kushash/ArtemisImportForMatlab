import numpy as np
from scipy.interpolate import interp1d

def resample_signal(t, sig, fs):
    t_new = np.linspace(t[0], t[-1], int((t[-1]-t[0])*fs))
    return t_new, interp1d(t, sig, fill_value="extrapolate")(t_new)