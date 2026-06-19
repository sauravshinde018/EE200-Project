import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import json
from collections import Counter
import random

# Import your custom functions (Ensure these files are in the same folder!)
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes

# --- 1. PAGE CONFIG & NEON STYLING ---
st.set_page_config(page_title="Zapptain America", layout="wide", page_icon="⚡")

# Inject custom CSS for Dark Neon styling, large fonts, and Hash Blocks
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Rajdhani:wght@500;700&display=swap');

    /* Global Dark Theme overrides */
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Neon Logo & Title */
    .logo-container {
        text-align: center;
        padding: 40px 0 20px 0;
    }
    .neon-title {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 4.5rem;
        color: #ffffff;
        text-shadow: 0 0 10px #00f3ff, 0 0 20px #00f3ff, 0 0 40px #00f3ff;
        margin-bottom: 0px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .neon-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        color: #ff007f;
        text-shadow: 0 0 5px #ff007f;
        letter-spacing: 4px;
        margin-top: 10px;
        margin-bottom: 50px;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 30px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.2rem;
        color: #00f3ff;
    }
    
    /* Hash Block UI (The Library) */
    .song-card {
        background: #1f2833;
        border: 2px solid #00f3ff;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .song-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        color: #39ff14;
        text-shadow: 0 0 8px #39ff14;
        margin-bottom: 15px;
    }
    .hash-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .hash-block {
        font-family: 'Rajdhani', monospace;
        font-weight: 700;
        font-size: 1rem;
        padding: 5px 10px;
        border-radius: 4px;
        background-color: rgba(0,0,0,0.5);
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: transparent !important;
        border: 2px solid #ff007f !important;
        color: #ff007f !important;
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        font-size: 1.2rem;
        width: 100%;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-shadow: 0 0 5px #ff007f;
        box-shadow: 0 0 10px rgba(255, 0, 127, 0.4) inset;
    }
    .stButton>button:hover {
        background-color: #ff007f !important;
        color: #fff !important;
        box-shadow: 0 0 20px #ff007f;
    }
    </style>
""", unsafe_allow_html=True)

# Set matplotlib to dark mode to match our neon theme
plt.style.use('dark_background')

# --- 2. DATABASE AND MATCHING LOGIC ---


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

    best_song, best_score, best_offset = None, 0, 0
    for song, offsets in matches_per_song.items():
        offset_counts = Counter(offsets)
        if offset_counts:
            most_common_offset, count = offset_counts.most_common(1)[0]
            if count > best_score:
                best_score = count
                best_song = song
                best_offset = most_common_offset

    return best_song, best_score, matches_per_song.get(best_song, [])


# --- 3. UI LAYOUT ---

# Neon Header / Logo
st.markdown("""
    <div class='logo-container'>
        <div class='neon-title'>⚡ ZAPPTAIN AMERICA</div>
        <div class='neon-subtitle'>NEURAL AUDIO FINGERPRINTING</div>
    </div>
""", unsafe_allow_html=True)

if not database:
    st.error("⚠️ DATABASE OFFLINE. Run `python build_db.py` to compile the databank.")

# Removed Batch Processing - Just two streamlined tabs now
tab1, tab2 = st.tabs(["[ THE VAULT ]", "[ INITIATE SCAN ]"])

# --- TAB 1: THE VAULT (Hash Library Visualizer) ---
with tab1:
    if database:
        # Re-organize database to map Songs -> Hashes for visualization
        song_data = {}
        for hash_str, occurrences in database.items():
            for occ in occurrences:
                song = occ['song']
                if song not in song_data:
                    song_data[song] = []
                # Keep up to 25 hashes per song for UI purposes so we don't crash the browser
                if len(song_data[song]) < 25:
                    song_data[song].append(hash_str)

        # Pre-defined neon colors for the hash blocks
        neon_colors = [
            {"border": "#00f3ff", "text": "#00f3ff",
                "glow": "rgba(0, 243, 255, 0.4)"},  # Cyan
            {"border": "#ff007f", "text": "#ff007f",
                "glow": "rgba(255, 0, 127, 0.4)"},  # Pink
            {"border": "#39ff14", "text": "#39ff14",
                "glow": "rgba(57, 255, 20, 0.4)"},  # Green
            {"border": "#fce803", "text": "#fce803",
                "glow": "rgba(252, 232, 3, 0.4)"}   # Yellow
        ]

        st.markdown(
            f"<p style='text-align:center; color:#c5c6c7; font-family:Rajdhani;'>INDEXED ENTITIES: {len(song_data)} | ACTIVE HASHES: {len(database):,}</p>", unsafe_allow_html=True)
        st.divider()

        # Render each song as a futuristic UI Card
        for song_name, hashes in sorted(song_data.items()):
            html_content = f"<div class='song-card'><div class='song-title'>[+] {song_name}</div><div class='hash-container'>"

            for h in hashes:
                color = random.choice(neon_colors)
                html_content += f"""
                    <div class='hash-block' style='
                        border: 1px solid {color["border"]}; 
                        color: {color["text"]}; 
                        box-shadow: 0 0 8px {color["glow"]};'>
                        {h}
                    </div>
                """
            html_content += "</div></div>"
            st.markdown(html_content, unsafe_allow_html=True)

# --- TAB 2: IDENTIFY (Single Clip Scan) ---
with tab2:
    st.markdown("<h3 style='font-family:Orbitron; color:#00f3ff; text-align:center;'>UPLOAD AUDIO FRAGMENT</h3>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "", type=['wav', 'mp3'], key="single_upload")

    if uploaded_file is not None:
        st.audio(uploaded_file)

        # Big Neon Button
        if st.button("RUN SPECTRAL ANALYSIS"):
            with st.spinner("Extracting Constellation Data..."):
                audio_data, fs = librosa.load(
                    uploaded_file, sr=None, mono=True)

                f, t, Sxx_db = get_spectrogram(audio_data, fs)
                t_frames, f_bins = get_constellation(Sxx_db)
                query_hashes = generate_hashes(t_frames, f_bins)

                # Visuals
                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        "<h4 style='font-family:Rajdhani; color:#ff007f;'>[ SPECTROGRAM ]</h4>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # Switched to Plasma for neon feel
                    ax.pcolormesh(
                        t, f, Sxx_db, shading='gouraud', cmap='plasma')
                    ax.set_ylim(0, 5000)
                    ax.axis('off')
                    st.pyplot(fig)

                with col2:
                    st.markdown(
                        "<h4 style='font-family:Rajdhani; color:#39ff14;'>[ CONSTELLATION MAP ]</h4>", unsafe_allow_html=True)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    # Plot points in neon cyan
                    ax.scatter(t_frames, f_bins, s=5,
                               color='#00f3ff', alpha=0.9)
                    ax.set_ylim(0, 5000)
                    ax.axis('off')
                    st.pyplot(fig)

                # Results
                st.markdown(
                    "<h4 style='font-family:Rajdhani; color:#fce803; text-align:center; margin-top:30px;'>[ OFFSET ALIGNMENT ]</h4>", unsafe_allow_html=True)
                best_song, score, winning_offsets = find_match(
                    query_hashes, database)

                if best_song and score > 10:
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.hist(winning_offsets, bins=100, color='#39ff14',
                            alpha=0.8)  # Neon Green histogram
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#c5c6c7')
                    ax.spines['bottom'].set_color('#c5c6c7')
                    ax.tick_params(colors='#c5c6c7')
                    st.pyplot(fig)

                    st.markdown(f"""
                        <div style='background-color:#1f2833; border:2px solid #39ff14; padding:20px; border-radius:10px; text-align:center; box-shadow: 0 0 20px rgba(57, 255, 20, 0.3); margin-top:20px;'>
                            <h2 style='font-family:Orbitron; color:#39ff14; margin:0;'>MATCH CONFIRMED</h2>
                            <h1 style='font-family:Rajdhani; color:#fff; font-size:3rem; margin:10px 0;'>{best_song}</h1>
                            <p style='color:#c5c6c7; font-family:monospace; font-size:1.2rem; margin:0;'>Confidence Rating: {score}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()  # Added celebration
                else:
                    st.error("No definitive match found in the current database.")
