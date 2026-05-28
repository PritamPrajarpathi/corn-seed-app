import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json
import os
import pandas as pd
import altair as alt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AgriScan Dashboard",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'counts' not in st.session_state:
    st.session_state['counts'] = {'High': 0, 'Medium': 0, 'Low': 0}

# --- CHANGE HERE: Set Default to True ---
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = True 

# --- 3. THEME TOGGLE & STYLING ---

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    # The toggle will now start in the "On" position because the state is True
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state['dark_mode'], key="theme_toggle")
    st.session_state['dark_mode'] = dark_mode
    st.markdown("---")

# --- UPDATED LIGHT MODE CSS (Yellow Sidebar + Light Green Toggle) ---
LIGHT_MODE_CSS = """
<style>
    /* Main Background - Fresh Mint Gradient */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #e6f7ed 50%, #d1fae5 100%);
        color: #022c22;
    }
    
    /* GLOBAL TEXT OVERRIDES */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, span, div {
        color: #022c22 !important; /* Very Dark Green/Black */
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* --- TOGGLE BUTTON: LIGHT GREEN CUSTOMIZATION --- */
    div[data-testid="stToggle"] div[role="switch"] {
        border: 2px solid #4ade80 !important; /* Light Green Border */
    }
    /* When Active (Dark Mode ON) */
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #4ade80 !important; /* Light Green Fill */
    }
    /* When Inactive (Light Mode) */
    div[data-testid="stToggle"] div[role="switch"][aria-checked="false"] {
        background-color: #dcfce7 !important; /* Very Pale Green */
    }
    /* The sliding circle inside */
    div[data-testid="stToggle"] div[role="switch"] span {
        background-color: #ffffff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #d1fae5;
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #15803d !important;
        font-weight: 800 !important;
    }

    /* --- SIDEBAR: CORN SILK YELLOW THEME --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fefce8 0%, #fef9c3 100%); 
        border-right: 2px solid #eab308;
    }
    
    section[data-testid="stSidebar"] * {
        color: #0f172a !important; 
    }
    section[data-testid="stSidebar"] h3 {
        color: #854d0e !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #14532d !important;
        font-weight: 800;
        text-shadow: none;
    }
    
    /* Subtext */
    .small-muted {
        color: #475569 !important;
        font-size: 1.1rem;
        font-weight: 500;
    }

    /* Radio Buttons */
    div[data-testid="stRadio"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    /* Primary Button */
    div.stButton > button {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: #ffffff !important;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.3);
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #15803d, #14532d);
        transform: translateY(-2px);
    }

    /* File Uploader */
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
        background-color: #ffffff; 
        border: 2px dashed #16a34a; 
        padding: 20px;
    }
    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }
    div[data-testid="stFileUploader"] button {
        background-color: #dcfce7 !important; 
        color: #14532d !important;
        border: 1px solid #16a34a !important;
        font-weight: 700 !important;
    }

    /* Data Tables */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
</style>
"""

# --- DARK MODE CSS (UNCHANGED) ---
DARK_MODE_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
    }
    html, body, [class*="css"] {
        color: #e2e8f0 !important;
        font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    /* Custom Toggle Switch for Dark Mode too */
    div[data-testid="stToggle"] div[role="switch"] {
        border: 2px solid #4ade80 !important;
    }
    div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
        background-color: #4ade80 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 15px rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #22c55e !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(34, 197, 94, 0.3);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid rgba(34, 197, 94, 0.4);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    h1, h2, h3 {
        font-family: "Playfair Display", "Georgia", serif;
        color: #f1f5f9 !important;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(34, 197, 94, 0.2);
    }
    .small-muted {
        color: #94a3b8 !important;
        font-size: 1rem;
        font-weight: 500;
    }
    div[data-testid="stRadio"] label {
        color: #e2e8f0 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stRadio"] label span {
        color: #e2e8f0 !important;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: #ffffff !important;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.5), 0 0 30px rgba(34, 197, 94, 0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #16a34a, #15803d);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(34, 197, 94, 0.6), 0 0 40px rgba(34, 197, 94, 0.3);
    }
    div[data-testid="stFileUploader"] {
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.6);
        border: 2px dashed rgba(34, 197, 94, 0.5);
    }
    div[data-testid="stFileUploader"] button {
        color: #e2e8f0 !important;
        background-color: rgba(34, 197, 94, 0.2) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
    }
    div[data-testid="stFileUploader"] label {
        color: #94a3b8 !important;
    }
    .stDataFrame {
        background: rgba(30, 41, 59, 0.8) !important;
        border-radius: 8px;
    }
    .stDataFrame th {
        color: #22c55e !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
    }
    .stDataFrame td {
        color: #e2e8f0 !important;
        background-color: rgba(30, 41, 59, 0.6) !important;
    }
    .stMarkdown, .stMarkdown p {
        color: #cbd5e1 !important;
    }
</style>
"""

# Apply CSS based on state
if st.session_state['dark_mode']:
    st.markdown(DARK_MODE_CSS, unsafe_allow_html=True)
else:
    st.markdown(LIGHT_MODE_CSS, unsafe_allow_html=True)

# --- 4. BACKEND LOGIC (UNCHANGED) ---
@st.cache_resource
def load_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'corn_model.h5')
    json_path = os.path.join(base_dir, 'classes.json')

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        return None, None
        
    try:
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        label_map = {v: k for k, v in class_indices.items()}
    except:
        label_map = None
        
    return model, label_map

model, label_map = load_assets()

def process_and_predict(image_data, model):
    size = (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0
    img_reshape = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# --- 5. DASHBOARD UI LAYOUT ---

st.title("🌽 AgriScan Live Dashboard")
st.markdown("<p class='small-muted'>Real-time quality control analytics for corn seed batches.</p>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
total_scans = sum(st.session_state['counts'].values())

with col1:
    st.metric("Total Scanned", f"{total_scans}", delta=None)
with col2:
    st.metric("High Quality", f"{st.session_state['counts']['High']}", delta="Export Ready")
with col3:
    st.metric("Medium Quality", f"{st.session_state['counts']['Medium']}", delta="- Processing", delta_color="off")
with col4:
    st.metric("Low Quality", f"{st.session_state['counts']['Low']}", delta="- Rejected", delta_color="inverse")

st.markdown("---")

main_col_1, main_col_2 = st.columns([1, 2])

with main_col_1:
    st.subheader("🔍 Seed Scanner")
    if model is None:
        st.error("🚨 Model not found. Please upload 'corn_model.h5'.")
        st.stop()

    mode = st.radio("Input Mode", ["Upload", "Camera"], horizontal=True)
    file_input = None
    if mode == "Upload":
        file_input = st.file_uploader("Upload Image", type=["jpg", "png"], label_visibility="collapsed")
    else:
        file_input = st.camera_input("Capture")

    if file_input:
        image = Image.open(file_input)
        st.image(image, caption="Current Specimen", use_container_width=True)
        
        if st.button("Run Analysis", use_container_width=True):
            with st.spinner("Analyzing..."):
                preds = process_and_predict(image, model)
                result_idx = np.argmax(preds)
                confidence = np.max(preds) * 100
                
                if label_map:
                    raw_label = label_map[result_idx].lower()
                else:
                    raw_label = str(result_idx)

                if "healthy" in raw_label:
                    grade = "High"
                elif "discolored" in raw_label or "silkcut" in raw_label:
                    grade = "Medium"
                else:
                    grade = "Low"

                st.session_state['counts'][grade] += 1
                st.session_state['history'].append({
                    "Grade": grade,
                    "Confidence": f"{confidence:.1f}%",
                    "Time": pd.Timestamp.now().strftime("%H:%M:%S")
                })
                
                st.success(f"**Result: {grade} Quality** ({confidence:.1f}%)")

with main_col_2:
    st.subheader("📊 Batch Analytics")
    
    chart_data = pd.DataFrame([
        {"Quality": "High", "Count": st.session_state['counts']['High'], "Color": "#2ecc71"},
        {"Quality": "Medium", "Count": st.session_state['counts']['Medium'], "Color": "#f1c40f"},
        {"Quality": "Low", "Count": st.session_state['counts']['Low'], "Color": "#e74c3c"}
    ])

    if total_scans > 0:
        c = alt.Chart(chart_data).mark_bar().encode(
            x='Quality',
            y='Count',
            color=alt.Color('Quality', scale=alt.Scale(domain=['High', 'Medium', 'Low'], range=['#2ecc71', '#f1c40f', '#e74c3c'])),
            tooltip=['Quality', 'Count']
        ).properties(height=300)
        
        st.altair_chart(c, use_container_width=True)
        
        st.write("**Recent Scans Log:**")
        if len(st.session_state['history']) > 0:
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist.tail(5), use_container_width=True)
    else:
        st.info("Waiting for data... Scan a seed to see analytics.")
