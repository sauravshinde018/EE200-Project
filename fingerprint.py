import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter


def get_spectrogram(audio_data, fs=44100):
    f, t, Sxx = signal.spectrogram(audio_data, fs, nperseg=4096, noverlap=2048)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def get_constellation(Sxx_db, neighborhood_size=30, threshold_percentile=97):
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db
    threshold = np.percentile(Sxx_db, threshold_percentile)
    is_loud_enough = Sxx_db > threshold
    peaks = local_max & is_loud_enough
    freq_bins, time_frames = np.where(peaks)

    return time_frames, freq_bins
