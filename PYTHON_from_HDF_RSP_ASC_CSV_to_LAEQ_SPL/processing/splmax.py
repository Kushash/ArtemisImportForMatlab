import numpy as np
from scipy.signal import butter, sosfilt

P_REF = 2e-5

def compute_spl_max(signal, fs):
    center_freqs = 10**((np.arange(-30, 14)) / 10) * 1000
    center_freqs = center_freqs[(center_freqs >= 1) & (center_freqs <= fs/2.5)]

    def a_weight(f):
        f1,f2,f3,f4 = 20.6,107.7,737.9,12194
        return 20*np.log10((f**4*f4**2)/((f**2+f1**2)*np.sqrt((f**2+f2**2)*(f**2+f3**2))*(f**2+f4**2)))+2

    weighted = np.zeros_like(signal)

    for fc in center_freqs:
        f_low, f_high = fc/(2**(1/6)), fc*(2**(1/6))
        sos = butter(7, [f_low/(fs/2), f_high/(fs/2)], btype='band', output='sos')
        filtered = sosfilt(sos, signal)

        weighted += filtered * (10**(a_weight(fc)/20))

    spl = 20*np.log10(np.abs(weighted)/P_REF + 1e-12)
    return np.max(spl)