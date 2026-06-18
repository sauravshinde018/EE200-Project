import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import json
from collections import Counter
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes  # Import the hasher we just made

# Load Database once using Streamlit caching for speed


@st.cache_data
def load_database():
    with open("song_database.json", "r") as f:
        return json.load(f)


database = load_database()


def find_match(query_hashes, db):
    """Finds the song with the largest alignment spike."""
    matches_per_song = {}

    for query_hash, query_t in query_hashes:
        if query_hash in db:
            for match in db[query_hash]:
                song = match['song']
                db_t = match['time']

                # The crucial step: Calculate the time offset
                offset = db_t - query_t

                if song not in matches_per_song:
                    matches_per_song[song] = []
                matches_per_song[song].append(offset)

    # Find the song with the highest concentration of a single offset
    best_song = None
    best_score = 0
    best_offset = 0

    for song, offsets in matches_per_song.items():
        # Count how many times each offset occurs
        offset_counts = Counter(offsets)
        if offset_counts:
            most_common_offset, count = offset_counts.most_common(1)[0]
            if count > best_score:
                best_score = count
                best_song = song
                best_offset = most_common_offset

    return best_song, best_score, matches_per_song.get(best_song, [])

# --- Inside your Streamlit IDENTIFY TAB (tab2) ---


if uploaded_file is not None:
    st.success("File uploaded successfully! Processing...")

    fs, audio_data = wavfile.read(uploaded_file)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Extract features
    f, t, Sxx_db = get_spectrogram(audio_data, fs)
    t_frames, f_bins = get_constellation(Sxx_db)
    query_hashes = generate_hashes(t_frames, f_bins)

    # 1. Visualization Step
    st.markdown("### STEP 1 - FEATURE EXTRACTION")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spectrogram**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
        ax.set_ylim(0, 5000)  # Focus on lower frequencies
        st.pyplot(fig)

    with col2:
        st.markdown("**Constellation Peaks**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(t_frames, f_bins, s=5, color='cyan', alpha=0.8)
        ax.set_facecolor('black')
        ax.set_ylim(0, 5000)
        st.pyplot(fig)

    # 2. Matching Step
    st.markdown("### STEP 2 - THE PROOF (Alignment Spike)")
    best_song, score, winning_offsets = find_match(query_hashes, database)

    if best_song and score > 10:  # Threshold to prevent false positives
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(winning_offsets, bins=100, color='orange')
        ax.set_title(f"Offset Histogram for {best_song}")
        ax.set_xlabel("Time Offset")
        ax.set_ylabel("Number of Matched Hashes")
        st.pyplot(fig)

        st.success(
            f"🎉 MATCH FOUND: **{best_song}** (Confidence Score: {score})")
    else:
        st.error("No definitive match found in the database.")
