import streamlit as st
import pandas as pd
import numpy as np
import librosa
import json
import re
import gc
import time
from collections import Counter
import plotly.graph_objects as go

# Import your custom functions
from fingerprint import get_spectrogram, get_constellation
from build_db import generate_hashes

# Set page config
st.set_page_config(page_title="Zapptain America",
                   layout="wide", page_icon="🎶")

# --- CUSTOM CSS: THE CYBERPUNK THEME ---
st.markdown("""
<style>
    /* Global Fonts */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, .telemetry-value, .hash-count {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Section Headers */
    .step-header {
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #00FFFF;
        margin-bottom: -10px;
        margin-top: 30px;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(90deg, rgba(0,255,255,0.05) 0%, rgba(0,255,255,0.15) 50%, rgba(0,255,255,0.05) 100%);
        border-top: 1px solid #00FFFF;
        border-bottom: 1px solid #00FFFF;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
        margin: 0;
    }
    .hero-subtitle {
        color: #A0A0A0;
        font-size: 1rem;
        margin-top: 5px;
    }
    .highlight-score { color: #FFA500; font-weight: bold; }
    
    /* Telemetry Cards */
    .telemetry-box {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
    }
    .telemetry-title {
        color: #00FFFF;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .telemetry-value {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 5px 0;
    }
    .telemetry-sub {
        color: #666;
        font-size: 0.7rem;
    }
    
    /* Library Cards */
    .library-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .library-card:hover {
        transform: translateY(-2px);
        border-color: #00FFFF;
    }
    .library-card h4 {
        margin: 0 0 5px 0;
        font-size: 1.1rem;
        color: #e6edf3;
    }
    .hash-count {
        font-size: 0.85rem;
        color: #8b949e;
        margin: 0;
    }
    
    /* Leaderboard Bars */
    .leaderboard-row {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
    }
    .lb-name { width: 30%; color: #e6edf3; font-size: 0.9rem;}
    .lb-bar-container { width: 60%; background-color: #222; height: 8px; border-radius: 4px; overflow: hidden; margin: 0 15px; }
    .lb-bar { background-color: #00FFFF; height: 100%; }
    .lb-score { width: 10%; color: #8b949e; font-size: 0.9rem; text-align: right; font-family: 'JetBrains Mono', monospace;}
</style>
""", unsafe_allow_html=True)

# --- UTILITY: TITLE FORMATTER ---


def format_title(raw_name):
    if not raw_name or raw_name == "No_Match":
        return "No Match"
    pretty_name = re.sub(r"(?<=[a-zA-Z])_(?=[a-zA-Z])", "'", raw_name)
    return pretty_name.replace("_", " ")

# --- DATABASE LOGIC ---


@st.cache_resource
def load_database():
    try:
        with open("song_database.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


database = load_database()


def find_match_leaderboard(query_hashes, db):
    """Returns top candidates instead of just the best song."""
    matches_per_song = {}
    for query_hash, query_t in query_hashes:
        if query_hash in db:
            for match in db[query_hash]:
                song, db_t = match['song'], match['time']
                offset = db_t - query_t
                if song not in matches_per_song:
                    matches_per_song[song] = []
                matches_per_song[song].append(offset)

    candidates = []
    for song, offsets in matches_per_song.items():
        offset_counts = Counter(offsets)
        if offset_counts:
            best_offset, count = offset_counts.most_common(1)[0]
            candidates.append({
                'song': song,
                'score': count,
                'offset': best_offset,
                'all_offsets': offsets
            })

    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def get_full_song_constellation(db, song_name):
    """Reconstructs the constellation of a full song from the stored hashes."""
    t_vals, f_vals = [], []
    for hash_str, matches in db.items():
        for match in matches:
            if match['song'] == song_name:
                f1 = int(hash_str.split('|')[0])
                t_vals.append(match['time'])
                f_vals.append(f1)
    return t_vals, f_vals


# --- UI LAYOUT ---

st.title("🎶 Zapptain America")
st.markdown("### **EE200: Audio Fingerprinting**")
st.write("Index a library of songs as spectrogram fingerprints, then identify any short clip against it.")
st.divider()

if not database:
    st.error("⚠️ **Database not found!** Please run `python build_db.py` locally and upload `song_database.json`.")

tab1, tab2, tab3 = st.tabs(["📚 LIBRARY", "🔍 IDENTIFY", "📋 BATCH"])

# --- TAB 1: LIBRARY ---
with tab1:
    st.markdown("<p class='step-header'>Indexed Database</p>",
                unsafe_allow_html=True)
    st.info("Song indexing is managed by the admin. The database is pre-loaded with the provided song library.")

    if database:
        song_counts = Counter()
        for hash_matches in database.values():
            for match in hash_matches:
                song_counts[match['song']] += 1

        # CSS Grid display for Library
        cols = st.columns(4)
        sorted_songs = sorted(song_counts.items(), key=lambda x: x[0])

        for idx, (raw_name, count) in enumerate(sorted_songs):
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="library-card">
                    <h4>{format_title(raw_name)}</h4>
                    <p class="hash-count">{count:,} hashes</p>
                </div>
                """, unsafe_allow_html=True)

# --- TAB 2: IDENTIFY ---
with tab2:
    st.markdown("<p class='step-header'>Identify a Clip</p>",
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=[
                                     'wav', 'mp3'], key="single_upload", label_visibility="collapsed")

    if uploaded_file is not None:

        if st.button("🚀 IDENTIFY TRACK", type="primary", use_container_width=True):

            # --- START TELEMETRY ---
            t_start = time.time()

            # 1. Audio Loading
            t0 = time.time()
            audio_data, fs = librosa.load(uploaded_file, sr=22050, mono=True)
            t_load = (time.time() - t0) * 1000

            # 2. Spectrogram
            t0 = time.time()
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            t_spectro = (time.time() - t0) * 1000

            # 3. Constellation
            t0 = time.time()
            t_frames, f_bins = get_constellation(Sxx_db)
            t_const = (time.time() - t0) * 1000

            # 4. Hashing
            t0 = time.time()
            query_hashes = generate_hashes(t_frames, f_bins)
            t_hash = (time.time() - t0) * 1000

            # 5. DB Lookup & Scoring
            t0 = time.time()
            candidates = find_match_leaderboard(query_hashes, database)
            t_db = (time.time() - t0) * 1000

            t_total = (time.time() - t_start) * 1000

            total_query_hashes = len(query_hashes)
            winner = candidates[0] if candidates else None
            is_match = winner and winner['score'] >= 5

            # --- TELEMETRY DASHBOARD ---
            m1, m2, m3, m4, m5, m6 = st.columns(6)

            m1.markdown(
                f"<div class='telemetry-box'><div class='telemetry-title'>① Spectrogram</div><div class='telemetry-value'>{int(t_spectro)} ms</div><div class='telemetry-sub'>{len(f)}x{len(t)} array</div></div>", unsafe_allow_html=True)
            m2.markdown(
                f"<div class='telemetry-box'><div class='telemetry-title'>② Constellation</div><div class='telemetry-value'>{int(t_const)} ms</div><div class='telemetry-sub'>{len(t_frames)} peaks</div></div>", unsafe_allow_html=True)
            m3.markdown(
                f"<div class='telemetry-box'><div class='telemetry-title'>③ Hashing</div><div class='telemetry-value'>{int(t_hash)} ms</div><div class='telemetry-sub'>{total_query_hashes:,} hashes</div></div>", unsafe_allow_html=True)
            m4.markdown(
                f"<div class='telemetry-box'><div class='telemetry-title'>④ DB Lookup</div><div class='telemetry-value'>{int(t_db)} ms</div><div class='telemetry-sub'>All tracks</div></div>", unsafe_allow_html=True)
            m5.markdown(
                f"<div class='telemetry-box'><div class='telemetry-title'>⑤ Scoring</div><div class='telemetry-value'>--</div><div class='telemetry-sub'>Offset {winner['offset'] if is_match else 'N/A'}</div></div>", unsafe_allow_html=True)
            m6.markdown(
                f"<div style='text-align: right; padding-top: 15px; color: #00FFFF; font-weight: bold;'>total {int(t_total)} ms</div>", unsafe_allow_html=True)

            # --- MATCH REVEAL & LEADERBOARD ---
            if is_match:
                pretty_song = format_title(winner['song'])
                runner_up_score = candidates[1]['score'] if len(
                    candidates) > 1 else 1
                multiplier = winner['score'] / \
                    runner_up_score if runner_up_score > 0 else winner['score']

                st.markdown(f"""
                <div class="hero-banner">
                    <p style="color: #00FFFF; margin:0; font-size: 0.9rem; letter-spacing: 2px;">MATCH FOUND</p>
                    <h1 class="hero-title">{pretty_song}</h1>
                    <p class="hero-subtitle">cluster score <span class="highlight-score">{winner['score']}</span> • <span class="highlight-score">{multiplier:.1f}x</span> the runner-up</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(
                    "<p class='step-header'>Candidate Scores</p><br>", unsafe_allow_html=True)
                max_score = winner['score']
                for cand in candidates[:5]:  # Top 5
                    width = (cand['score'] / max_score) * 100
                    st.markdown(f"""
                    <div class="leaderboard-row">
                        <div class="lb-name">{format_title(cand['song'])}</div>
                        <div class="lb-bar-container"><div class="lb-bar" style="width: {width}%;"></div></div>
                        <div class="lb-score">{cand['score']}</div>
                    </div>
                    """, unsafe_allow_html=True)

            else:
                st.error(
                    "❌ No definitive match found. The confidence score was too low.")
                st.stop()

            st.divider()

            # --- VIZ 1: SPECTROGRAM & CONSTELLATION (PLOTLY) ---
            st.markdown(
                "<p class='step-header'>STEP 1 • FEATURE EXTRACTION</p>", unsafe_allow_html=True)
            st.markdown("### From spectrogram to constellation")
            st.markdown(
                f"<p style='color: #8b949e;'>The clip was converted into a time-frequency map (left). From that rich image, only the <b>{len(t_frames)} most prominent peaks</b> were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</p>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)

            # Plotly Spectrogram
            fig_spec = go.Figure(data=go.Heatmap(
                z=Sxx_db, x=t, y=f, colorscale='Magma', showscale=False))
            fig_spec.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                xaxis=dict(title='time (s)', color='white', showgrid=False),
                yaxis=dict(title='frequency (Hz)', color='white',
                           range=[0, 4000], showgrid=False)
            )
            c1.plotly_chart(fig_spec, use_container_width=True)

            # Plotly Constellation
            fig_const = go.Figure(data=go.Scatter(
                x=t_frames, y=f_bins, mode='markers', marker=dict(color='#00FFFF', size=3, opacity=0.7)))
            fig_const.update_layout(
                plot_bgcolor='#0e1117', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                xaxis=dict(title='time (frames)', color='white',
                           showgrid=False, zeroline=False),
                yaxis=dict(title='freq bin', color='white', range=[
                           0, len(f)//3], showgrid=False, zeroline=False)
            )
            c2.plotly_chart(fig_const, use_container_width=True)

            st.divider()

            # --- VIZ 2: WHERE IN THE SONG? ---
            st.markdown(
                "<p class='step-header'>STEP 2 • DATABASE SEARCH</p>", unsafe_allow_html=True)
            st.markdown("### Where in the song?")

            # Reconstruct the full song fingerprint
            full_t, full_f = get_full_song_constellation(
                database, winner['song'])
            query_len_frames = max(t_frames) if len(t_frames) > 0 else 0

            st.markdown(f"<p style='color: #8b949e;'>The <b>{total_query_hashes:,} fingerprint hashes</b> were looked up against every indexed track. Below is the full fingerprint of <i>{pretty_song}</i> reconstructed from the database. The highlighted window is exactly where the query clip sits inside the full song.</p>", unsafe_allow_html=True)

            fig_where = go.Figure(data=go.Scatter(
                x=full_t, y=full_f, mode='markers', marker=dict(color='#00FFFF', size=2, opacity=0.5)))

            # Add highlight bounding box
            fig_where.add_vrect(x0=winner['offset'], x1=winner['offset'] + query_len_frames,
                                fillcolor="#00FFFF", opacity=0.2, line_width=1, line_color="#00FFFF")

            fig_where.update_layout(
                plot_bgcolor='#0e1117', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=350,
                xaxis=dict(title='time (frames)', color='white',
                           showgrid=False, zeroline=False),
                yaxis=dict(title='freq bin', color='white', range=[0, max(
                    full_f) if full_f else 500], showgrid=False, zeroline=False)
            )
            st.plotly_chart(fig_where, use_container_width=True)

            st.divider()

            # --- VIZ 3: ALIGNMENT SPIKE ---
            st.markdown(
                "<p class='step-header'>STEP 3 • THE PROOF</p>", unsafe_allow_html=True)
            st.markdown("### The alignment spike")
            st.markdown(
                f"<p style='color: #8b949e;'>Every matched hash votes for a time offset. Chance matches scatter randomly, forming a flat noise floor. A genuine match makes them converge: <b style='color:#FFA500'>{winner['score']} hashes agreed on a single offset.</b> That spike cannot be a coincidence.</p>", unsafe_allow_html=True)

            hist_data = winner['all_offsets']

            # Create histogram
            fig_hist = go.Figure()
            # Plot the flat noise floor (all data)
            fig_hist.add_trace(go.Histogram(
                x=hist_data, nbinsx=150, marker_color='#1a535c', name="Noise"))

            # Highlight the winning bin by overlaying a bar exactly at the offset
            fig_hist.add_trace(go.Bar(x=[winner['offset']], y=[winner['score']], width=np.ptp(
                hist_data)/150, marker_color='#FFA500', name="Match"))

            # Annotation Arrow
            fig_hist.add_annotation(x=winner['offset'], y=winner['score'], text=f"{winner['score']} hashes<br>align here", showarrow=True,
                                    arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#FFA500", font=dict(color="#FFA500"), ax=50, ay=-50)

            fig_hist.update_layout(
                barmode='overlay', showlegend=False,
                plot_bgcolor='#0e1117', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=20, b=0), height=350,
                xaxis=dict(title='time offset (database frame - query frame)',
                           color='white', showgrid=False, zeroline=False),
                yaxis=dict(title='# hashes', color='white',
                           showgrid=True, gridcolor='#333', zeroline=False)
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Cleanup
            del audio_data, Sxx_db, t_frames, f_bins, query_hashes
            gc.collect()

# --- TAB 3: BATCH MODE ---
with tab3:
    st.markdown("<p class='step-header'>Batch Processing</p>",
                unsafe_allow_html=True)
    st.write("Upload a set of query clips. Each is identified against the indexed library, and results are written to a standardized `results.csv`.")

    batch_files = st.file_uploader("Upload multiple audio files", type=[
                                   'wav', 'mp3'], accept_multiple_files=True, key="batch_upload")

    if st.button("▶️ RUN BATCH ANALYSIS", type="primary") and batch_files:
        results = []
        progress_bar = st.progress(
            0, text="Analyzing audio files. Please wait...")

        for i, file in enumerate(batch_files):
            audio_data, fs = librosa.load(file, sr=22050, mono=True)
            f, t, Sxx_db = get_spectrogram(audio_data, fs)
            t_frames, f_bins = get_constellation(Sxx_db)
            query_hashes = generate_hashes(t_frames, f_bins)

            candidates = find_match_leaderboard(query_hashes, database)
            best_song = candidates[0]['song'] if candidates else None
            score = candidates[0]['score'] if candidates else 0

            raw_prediction = best_song if (
                best_song and score >= 5) else "No_Match"

            results.append({
                "#": i + 1,
                "Audio File": file.name,
                "Identified Track": format_title(raw_prediction),
                "filename": file.name.rsplit('.', 1)[0],
                "prediction": raw_prediction
            })

            progress_bar.progress(
                (i + 1) / len(batch_files), text=f"Processed {i+1} of {len(batch_files)} files...")

            del audio_data, Sxx_db, t_frames, f_bins, query_hashes
            gc.collect()

        st.success("✅ **Batch Processing Complete!**")
        df_results = pd.DataFrame(results)

        st.dataframe(df_results[["#", "Audio File", "Identified Track"]],
                     hide_index=True, use_container_width=True)

        csv = df_results[["filename", "prediction"]].to_csv(
            index=False).encode('utf-8')
        st.download_button(label="⬇️ Download `results.csv`", data=csv,
                           file_name='results.csv', mime='text/csv', type="primary")
