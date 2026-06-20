import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
import re
from collections import Counter

# Import your custom functions
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes

# Set page config
st.set_page_config(page_title="Zapptain America",
                   layout="wide", page_icon="🎶")

# --- UTILITY: TITLE FORMATTER ---


def format_title(raw_name):
    """
    Magically cleans up filenames for the UI without altering the raw data.
    Fixes apostrophes (Can_t -> Can't) and replaces remaining underscores with spaces.
    """
    if not raw_name or raw_name == "No_Match":
        return "No Match"
    # Replace underscores between letters with an apostrophe
    pretty_name = re.sub(r"(?<=[a-zA-Z])_(?=[a-zA-Z])", "'", raw_name)
    # Replace any leftover underscores with spaces
    return pretty_name.replace("_", " ")

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

# App Header
st.title("🎶 Zapptain America")
st.markdown("### **EE200: Sonic Signatures & Audio Fingerprinting**")
st.write("Upload a short audio clip, and the system will identify the track using spectrogram constellations and hash alignment. 🚀")
st.divider()

if not database:
    st.error("⚠️ **Database not found!** Please run `python build_db.py` locally and upload `song_database.json` to GitHub.")

# Tabs
tab1, tab2, tab3 = st.tabs(
    ["📚 THE LIBRARY", "🔍 IDENTIFY CLIP", "📋 BATCH PROCESSING"])

# --- TAB 1: LIBRARY ---
with tab1:
    st.markdown("### 🗄️ Indexed Database")
    st.info("Song indexing is managed by the admin. The database is pre-loaded with the provided song library.")

    if database:
        st.success(
            f"✅ **Database loaded successfully!** The system is currently tracking **{len(database):,}** unique hash patterns.")

        st.markdown("#### 🎵 Tracks in Library")

        # Calculate how many hashes belong to each song
        song_counts = Counter()
        for hash_matches in database.values():
            for match in hash_matches:
                song_counts[match['song']] += 1

        # Build list using the pretty formatted titles
        formatted_counts = []
        for raw_name, count in song_counts.items():
            formatted_counts.append({"Track Name": format_title(
                raw_name), "Total Hashes Indexed": count})

        if formatted_counts:
            df_songs = pd.DataFrame(formatted_counts)
            df_songs = df_songs.sort_values(
                by="Track Name").reset_index(drop=True)
            df_songs.index = df_songs.index + 1
            st.dataframe(df_songs, use_container_width=True)

# --- TAB 2: IDENTIFY (Single Clip Mode) ---
with tab2:
    st.markdown("### 🎧 Identify a Mystery Clip")

    uploaded_file = st.file_uploader("Drop an audio file here (.wav or .mp3)", type=[
                                     'wav', 'mp3'], key="single_upload")

    if uploaded_file is not None:
        st.markdown("**Listen to your upload:**")
        st.audio(uploaded_file)

        if st.button("🚀 Identify Track", type="primary"):
            st.success("⏳ Analyzing the audio signal... Please wait.")

            audio_data, fs = librosa.load(uploaded_file, sr=22050, mono=True)
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            query_hashes = generate_hashes(t_frames, f_bins)
            total_query_hashes = len(query_hashes)

            st.divider()

            # 1. Feature Visualization
            st.markdown("### 🔬 STEP 1: Feature Extraction")
            st.write(
                "Converting the audio waveform into a time-frequency map, then isolating the most prominent peaks.")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🌊 Spectrogram")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='magma')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlabel('Time (s)')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                st.pyplot(fig)

            with col2:
                st.markdown("#### ✨ Constellation Map")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(t_frames, f_bins, s=5, color='#00FFFF', alpha=0.8)
                ax.set_ylabel('Frequency Bin')
                ax.set_xlabel('Time Frame')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('#1E1E1E')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                st.pyplot(fig)

            st.divider()

            # 2. Matching Visualization
            st.markdown("### 🎯 STEP 2: The Proof (Alignment Spike)")
            best_song, score, winning_offsets = find_match(
                query_hashes, database)

            if best_song and score >= 5:
                percentage = min((score / total_query_hashes) *
                                 100, 100.0) if total_query_hashes > 0 else 0.0

                # Formatted Title used for the UI Reveal!
                pretty_song_name = format_title(best_song)
                st.markdown(
                    f"<h2 style='text-align: center; color: #4CAF50;'>🎉 MATCH FOUND: {pretty_song_name}</h2>", unsafe_allow_html=True)

                met1, met2, met3 = st.columns(3)
                met1.metric(label="Confidence", value=f"{percentage:.1f}%")
                met2.metric(label="Aligned Hashes",
                            value=f"{score} / {total_query_hashes}")
                met3.metric(label="Library Tracks Searched", value="All")

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.hist(winning_offsets, bins=100,
                        color='#FFA500', edgecolor='black')
                ax.set_title(
                    f"Offset Histogram for {pretty_song_name}", color='white')
                ax.set_xlabel("Time Offset Difference", color='white')
                ax.set_ylabel("Matched Hashes", color='white')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('#1E1E1E')
                ax.tick_params(colors='white')
                st.pyplot(fig)

            else:
                st.error("❌ **No definitive match found.**")

# --- TAB 3: BATCH MODE ---
with tab3:
    st.markdown("### 🚀 Identify Many Clips at Once")
    st.write("Upload a set of query clips. Each is identified against the indexed library, and results are written to a standardized `results.csv`.")

    batch_files = st.file_uploader("Upload multiple audio files", type=[
                                   'wav', 'mp3'], accept_multiple_files=True, key="batch_upload")

    if st.button("▶️ Run Batch Analysis", type="primary") and batch_files:
        results = []
        progress_text = "Analyzing audio files. Please wait..."
        progress_bar = st.progress(0, text=progress_text)

        for i, file in enumerate(batch_files):
            audio_data, fs = librosa.load(file, sr=22050, mono=True)

            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            query_hashes = generate_hashes(t_frames, f_bins)

            best_song, score, _ = find_match(query_hashes, database)

            raw_prediction = best_song if (
                best_song and score >= 5) else "No_Match"

            # Save the beautiful names for the screen, and the raw names for the CSV
            results.append({
                "#": i + 1,
                "Audio File": file.name,
                "Identified Track": format_title(raw_prediction),
                "filename": file.name.rsplit('.', 1)[0],
                "prediction": raw_prediction
            })

            progress_bar.progress(
                (i + 1) / len(batch_files), text=f"Processed {i+1} of {len(batch_files)} files...")

        st.success("✅ **Batch Processing Complete!**")

        df_results = pd.DataFrame(results)
        st.markdown("#### 📊 Results Preview")

        # UI Table: Only show the pretty columns
        st.dataframe(df_results[["#", "Audio File", "Identified Track"]],
                     hide_index=True, use_container_width=True)

        # Download CSV: Only export the raw filename and prediction columns!
        csv = df_results[["filename", "prediction"]].to_csv(
            index=False).encode('utf-8')

        st.download_button(
            label="⬇️ Download `results.csv`",
            data=csv,
            file_name='results.csv',
            mime='text/csv',
            type="primary"
        )
