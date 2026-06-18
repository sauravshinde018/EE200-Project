import os
import json
import numpy as np
from scipy.io import wavfile
# Assuming you saved the functions from Part 2 in fingerprint.py
from fingerprint import get_spectrogram, get_constellation


def generate_hashes(time_frames, freq_bins, delay_window=15):
    """Pairs nearby peaks to create robust (f1, f2, delta_t) hashes."""
    hashes = []
    num_peaks = len(time_frames)

    # Sort peaks chronologically
    sort_idx = np.argsort(time_frames)
    t_sorted = time_frames[sort_idx]
    f_sorted = freq_bins[sort_idx]

    for i in range(num_peaks):
        # Look ahead a few peaks to form pairs
        for j in range(1, delay_window):
            if (i + j) < num_peaks:
                t1, f1 = int(t_sorted[i]), int(f_sorted[i])
                t2, f2 = int(t_sorted[i+j]), int(f_sorted[i+j])

                delta_t = t2 - t1
                # Only pair peaks if they are reasonably close in time
                if 0 < delta_t < 100:
                    # Create a string hash: "f1|f2|delta_t"
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    hashes.append((hash_str, t1))
    return hashes


def build_database(song_folder, output_file="song_database.json"):
    print("Building database... This might take a few minutes.")
    database = {}  # The inverted index

    for filename in os.listdir(song_folder):
        if filename.endswith(".wav"):  # Add .mp3 if using librosa instead of scipy
            song_name = filename.rsplit('.', 1)[0]
            print(f"Processing: {song_name}")

            filepath = os.path.join(song_folder, filename)
            fs, audio_data = wavfile.read(filepath)

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Extract features
            _, _, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            hashes = generate_hashes(t_frames, f_bins)

            # Populate the inverted index
            for hash_str, t1 in hashes:
                if hash_str not in database:
                    database[hash_str] = []
                database[hash_str].append({"song": song_name, "time": t1})

    # Save to disk
    with open(output_file, "w") as f:
        json.dump(database, f)
    print(f"✅ Database saved to {output_file}!")


# Run the builder (Make sure you have a folder named 'song_database' with your audio files)
if __name__ == "__main__":
    build_database("song_database")
