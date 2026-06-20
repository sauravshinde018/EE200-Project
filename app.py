import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
from collections import Counter

# Import your custom functions
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes

# Set page config
st.set_page_config(page_title="EE200: Audio Fingerprinting",
                   layout="wide", page_icon="🎵")

# --- DATABASE AND MATCHING LOGIC ---


@st.cache_data
def load_database():
    try:
        with open("song_database.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


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

    best_song = None
    best_score = 0
    best_offset = 0

    for song, offsets in matches_per_song.items():
        offset_counts = Counter(offsets)
        if offset_counts:
            most_common_offset, count = offset_counts.most_common(1)[0]
            if count > best_score:
                best_score = count
                best_song = song
                best_offset = most_common_offset

    return best_song, best_score, matches_per_song.get(best_song, [])

# --- UI LAYOUT ---


st.title("🎵 EE200: Audio Fingerprinting")
st.markdown(
    "Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")

if not database:
    st.error("⚠️ Database not found! Please run `python build_db.py` locally and upload `song_database.json` to GitHub.")

tab1, tab2, tab3 = st.tabs(["📚 LIBRARY", "🔍 IDENTIFY", "📋 BATCH"])

# --- TAB 1: LIBRARY ---
with tab1:
    st.markdown("### The Database")
    st.info("Song indexing is managed by the admin. The database is pre-loaded with the provided song library.")
    if database:
        st.success(
            f"Database loaded successfully! Contains {len(database)} unique hashes.")

# --- TAB 2: IDENTIFY (Single Clip Mode) ---
with tab2:
    st.markdown("### Identify a clip")

    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=[
                                     'wav', 'mp3'], key="single_upload")

    if uploaded_file is not None:
        st.success("File uploaded! Processing... Please wait.")

        # CRITICAL FIX: Force sr=22050 to match the database exactly
        audio_data, fs = librosa.load(uploaded_file, sr=22050, mono=True)

        # Extract features
        f, t, Sxx_db = get_spectrogram(audio_data, fs)
        t_frames, f_bins = get_constellation(Sxx_db)
        query_hashes = generate_hashes(t_frames, f_bins)

        # 1. Feature Visualization
        st.markdown("### STEP 1 - FEATURE EXTRACTION")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Spectrogram**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
            st.pyplot(fig)

        with col2:
            st.markdown("**Constellation Peaks**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(t_frames, f_bins, s=5, color='cyan', alpha=0.8)
            ax.set_facecolor('black')
            st.pyplot(fig)

        # 2. Matching Visualization
        st.markdown("### STEP 2 - THE PROOF (Alignment Spike)")
        best_song, score, winning_offsets = find_match(query_hashes, database)

        # Lowered threshold to 5 for better detection of short clips
        if best_song and score >= 5:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(winning_offsets, bins=100, color='orange')
            ax.set_title(f"Offset Histogram for {best_song}")
            ax.set_xlabel("Time Offset")
            ax.set_ylabel("Number of Matched Hashes")
            st.pyplot(fig)

            st.success(
                f"🎉 MATCH FOUND: **{best_song}** (Confidence Score: {score})")
        else:
            st.error("No definitive match found. Confidence score was too low.")

# --- TAB 3: BATCH MODE ---
with tab3:
    st.markdown("### Identify many clips at once")

    batch_files = st.file_uploader("Upload multiple files", type=[
                                   'wav', 'mp3'], accept_multiple_files=True, key="batch_upload")

    if st.button("Run Batch") and batch_files:
        results = []
        progress_bar = st.progress(0)

        for i, file in enumerate(batch_files):
            audio_data, fs = librosa.load(file, sr=22050, mono=True)

            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            query_hashes = generate_hashes(t_frames, f_bins)

            best_song, score, _ = find_match(query_hashes, database)

            prediction = best_song if (
                best_song and score >= 5) else "No_Match"
            results.append({"filename": file.name.rsplit(
                '.', 1)[0], "prediction": prediction})

            progress_bar.progress((i + 1) / len(batch_files))

        df_results = pd.DataFrame(results)
        st.markdown("### Results")
        st.dataframe(df_results)

        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download results.csv",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
        )
