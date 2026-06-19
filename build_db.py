import os
import json
import numpy as np
import librosa
from fingerprint import get_spectrogram, get_constellation


def generate_hashes(time_frames, freq_bins, delay_window=8):
    hashes = []
    num_peaks = len(time_frames)

    sort_idx = np.argsort(time_frames)
    t_sorted = time_frames[sort_idx]
    f_sorted = freq_bins[sort_idx]

    for i in range(num_peaks):
        for j in range(1, delay_window):
            if (i + j) < num_peaks:
                t1, f1 = int(t_sorted[i]), int(f_sorted[i])
                t2, f2 = int(t_sorted[i+j]), int(f_sorted[i+j])
                delta_t = t2 - t1

                if 0 < delta_t < 100:
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    hashes.append((hash_str, t1))

    return hashes


def build_database(song_folder, output_file="song_database.json"):
    print("Building database... This might take a few minutes.")
    database = {}
    songs_processed = 0

    for filename in os.listdir(song_folder):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            song_name = filename.rsplit('.', 1)[0]
            print(f"Processing: {song_name}...")

            filepath = os.path.join(song_folder, filename)
            audio_data, fs = librosa.load(filepath, sr=None, mono=True)

            _, _, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            hashes = generate_hashes(t_frames, f_bins)

            for hash_str, t1 in hashes:
                if hash_str not in database:
                    database[hash_str] = []
                database[hash_str].append({"song": song_name, "time": t1})

            songs_processed += 1

    with open(output_file, "w") as f:
        json.dump(database, f)

    print(f"✅ DONE! Processed {songs_processed} songs.")
    print(f"✅ Database saved to {output_file}!")


if __name__ == "__main__":
    build_database("song_database")
