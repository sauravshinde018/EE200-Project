import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter


def get_spectrogram(audio_data, fs=44100):
    """Computes the spectrogram of the audio signal."""
    # A balanced window size (e.g., 4096 samples) for good time/freq resolution
    f, t, Sxx = signal.spectrogram(audio_data, fs, nperseg=4096, noverlap=2048)

    # Convert to dB scale
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def get_constellation(Sxx_db, neighborhood_size=30, threshold_percentile=97):
    """
    Finds the 'constellation' of peaks (local maxima) in the spectrogram.
    Increased neighborhood and threshold to dramatically reduce file size.
    """
    # 1. Find local maxima using a 2D max filter
    local_max = maximum_filter(Sxx_db, size=neighborhood_size) == Sxx_db

    # 2. Apply a strict threshold to ignore quiet background noise
    threshold = np.percentile(Sxx_db, threshold_percentile)
    is_loud_enough = Sxx_db > threshold

    # 3. The peaks are where the signal is BOTH a local maximum AND loud enough
    peaks = local_max & is_loud_enough

    # Get the row (freq) and column (time) indices of the peaks
    freq_bins, time_frames = np.where(peaks)

    return time_frames, freq_bins
