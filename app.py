import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
from collections import Counter
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes

st.set_page_config(page_title="Zapptain America",
                   layout="centered", page_icon="⚡")

st.markdown("""
    <style>
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        text-align: center;
        letter-spacing: -1.5px;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 40px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 20px;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_database():
    try:
        with open("song_database.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


database = load_database()


def find_match(query_hashes, db):
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


st.markdown("<div class='main-title'>⚡ Zapptain America</div>",
            unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Acoustic Fingerprinting & Recognition</div>",
            unsafe_allow_html=True)

if not database:
    st.error("⚠️ Database not found! Please run `python build_db.py` locally and upload `song_database.json` to GitHub.")

tab1, tab2, tab3 = st.tabs(["📚 LIBRARY", "🔍 IDENTIFY", "📋 BATCH"])

with tab1:
    st.markdown("<h4 style='text-align:center;'>Indexed Database</h4>",
                unsafe_allow_html=True)

    if database:
        unique_songs = set()
        for hash_val, matches in database.items():
            for match in matches:
                unique_songs.add(match['song'])

        st.success(
            f"Database active. Contains **{len(database):,}** unique hashes across **{len(unique_songs)}** tracks.")

        df_songs = pd.DataFrame(
            sorted(list(unique_songs)), columns=["Track Name"])
        st.dataframe(df_songs, use_container_width=True, hide_index=True)
    else:
        st.info(
            "Song indexing is managed by the admin. The database is currently empty.")

with tab2:
    st.markdown("<h4 style='text-align:center;'>Upload & Analyze</h4>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Select an audio clip (.wav or .mp3)", type=[
                                     'wav', 'mp3'], key="single_upload", label_visibility="collapsed")

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("⚡ Identify Track"):
            with st.spinner("Analyzing spectral frequencies..."):
                audio_data, fs = librosa.load(
                    uploaded_file, sr=None, mono=True)

                f, t, Sxx_db = get_spectrogram(audio_data, fs)
                t_frames, f_bins = get_constellation(Sxx_db)
                query_hashes = generate_hashes(t_frames, f_bins)

                st.divider()
                st.markdown("##### 1. Extracted Features")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("*Spectrogram*")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.pcolormesh(
                        t, f, Sxx_db, shading='gouraud', cmap='magma')
                    ax.set_ylim(0, 5000)
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.markdown("*Constellation Map*")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(t_frames, f_bins, s=5, color='cyan', alpha=0.8)
                    ax.set_facecolor('#1e1e1e')
                    ax.set_ylim(0, 5000)
                    ax.axis('off')
                    st.pyplot(fig)

                st.markdown("##### 2. Offset Histogram")
                best_song, score, winning_offsets = find_match(
                    query_hashes, database)

                if best_song and score > 10:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.hist(winning_offsets, bins=100,
                            color='#00ff88', alpha=0.8)
                    ax.set_facecolor('#1e1e1e')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)

                    st.success(
                        f"### MATCH: {best_song} \n Confidence Score: {score}")
                else:
                    st.error("No definitive match found in the current database.")

with tab3:
    st.markdown("<h4 style='text-align:center;'>Batch Processor</h4>",
                unsafe_allow_html=True)
    batch_files = st.file_uploader("Upload multiple clips", type=[
                                   'wav', 'mp3'], accept_multiple_files=True, key="batch_upload", label_visibility="collapsed")

    if batch_files:
        if st.button("Process Batch"):
            results = []
            progress_bar = st.progress(0)

            for i, file in enumerate(batch_files):
                audio_data, fs = librosa.load(file, sr=None, mono=True)
                f, t, Sxx_db = get_spectrogram(audio_data, fs)
                t_frames, f_bins = get_constellation(Sxx_db)
                query_hashes = generate_hashes(t_frames, f_bins)

                best_song, score, _ = find_match(query_hashes, database)
                prediction = best_song if (
                    best_song and score > 10) else "No_Match"
                results.append(
                    {"filename": file.name, "prediction": prediction})

                progress_bar.progress((i + 1) / len(batch_files))

            df_results = pd.DataFrame(results)
            st.divider()
            st.dataframe(df_results, use_container_width=True)

            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download results.csv",
                data=csv,
                file_name='results.csv',
                mime='text/csv',
                use_container_width=True
            )
