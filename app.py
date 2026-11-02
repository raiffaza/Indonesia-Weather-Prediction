import os
import datetime
import calendar
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from catboost import CatBoostRegressor
from PIL import Image

# Browser tab icon logic
try:
    icon_img = Image.open("frieren icon.jfif") 
except Exception:
    icon_img = "🌤️" 

st.set_page_config(
    page_title="IDN-CLIM // Surface Temperature Workbench",
    page_icon=icon_img, 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar toggle state initialization
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

# Bespoke Meteorological Chart Styling (Warm Cream & Sky Blue Theme)
plt.rcParams.update({
    "figure.facecolor": "#FCFBF7",
    "axes.facecolor": "#FCFBF7",
    "text.color": "#182633",
    "axes.labelcolor": "#3A4D5E",
    "xtick.color": "#52667A",
    "ytick.color": "#52667A",
    "axes.edgecolor": "#D5D0C1",
    "grid.color": "#E6E2D6",
    "font.sans-serif": ["Plus Jakarta Sans", "DejaVu Sans", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False
})

# Dynamic CSS based on sidebar state
css_base = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Base Canvas: Warm Alabaster Cream */
    .stApp {
        background-color: #F7F4EB !important;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #182633 !important;
    }

    code, pre, .mono-val {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hide Default Streamlit Chrome elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stToolbar"] { visibility: hidden !important; }
    
    /* Hide native sidebar toggles to prevent conflicts with custom buttons */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* Custom Sidebar Button Styling */
    div[data-testid="stButton"] > button {
        background-color: #EFECE1 !important;
        color: #182633 !important;
        border: 1px solid #D5D0C1 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #E5E1D4 !important;
        border-color: #B3D1E7 !important;
        color: #0F4469 !important;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
    }

    /* Sidebar: Warm Parchment Cream */
    section[data-testid="stSidebar"] {
        background-color: #EFECE1 !important;
        border-right: 1px solid #D5D0C1 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #182633 !important;
    }

    .sidebar-section-title {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #4A6984 !important;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #D5D0C1;
    }

    /* Sidebar Navigation Radio */
    div[data-testid="stRadio"] > div { gap: 0.25rem; }
    div[data-testid="stRadio"] label {
        background-color: transparent !important;
        padding: 0.55rem 0.75rem !important;
        border-radius: 4px !important;
        border: 1px solid transparent !important;
        transition: background-color 0.15s ease;
        cursor: pointer;
    }
    div[data-testid="stRadio"] label:hover { background-color: #E5E1D4 !important; }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background-color: #DFEEF8 !important;
        border: 1px solid #9BC2DF !important;
    }
    div[data-testid="stRadio"] label[data-checked="true"] p {
        color: #0F4469 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stRadio"] label > div:first-child { display: none !important; }

    /* Inputs & Dropdowns */
    div[data-baseweb="select"] * { color: #182633 !important; }
    .st-emotion-cache-1wmy9hl, .st-emotion-cache-1wmy9hl p { color: #182633 !important; }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stDateInput"] > div > div {
        background-color: #FCFBF7 !important;
        border: 1px solid #C9C3B2 !important;
        color: #182633 !important;
        border-radius: 4px !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] svg { fill: #3A4D5E !important; }
    ul[role="listbox"] {
        background-color: #FCFBF7 !important;
        border: 1px solid #C9C3B2 !important;
    }
    ul[role="listbox"] li { color: #182633 !important; }
    ul[role="listbox"] li:hover { background-color: #E4F0F8 !important; }

    /* Masthead Header */
    .masthead {
        background-color: #E4F0F8;
        border: 1px solid #B3D1E7;
        border-top: 3px solid #1E5D88;
        border-radius: 4px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.75rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .masthead-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #C5DDF0;
        padding-bottom: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .masthead-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #1E5D88;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .masthead-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F293D !important;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .masthead-desc {
        font-size: 0.92rem;
        color: #2C465B !important;
        margin: 0;
        line-height: 1.55;
        max-width: 960px;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FCFBF7;
        border: 1px solid #D5D0C1;
        border-top: 2px solid #7BAFD4;
        border-radius: 4px;
        padding: 1.1rem 1.25rem;
        height: 100%;
    }
    .kpi-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #5A6D7C !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F293D !important;
        margin-bottom: 0.4rem;
        letter-spacing: -0.03em;
    }
    .kpi-delta-pos {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        color: #9E2A2B !important;
        background: #F9E6E6;
        border: 1px solid #E5B7B7;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        display: inline-block;
    }
    .kpi-delta-neg {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 600;
        color: #165682 !important;
        background: #E4F0F8;
        border: 1px solid #B3D1E7;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        display: inline-block;
    }
    .kpi-delta-neutral {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        font-weight: 500;
        color: #3E5060 !important;
        background: #EFECE1;
        border: 1px solid #D5D0C1;
        padding: 0.15rem 0.5rem;
        border-radius: 3px;
        display: inline-block;
    }

    /* Tech Panels */
    .tech-panel {
        background: #FCFBF7;
        border: 1px solid #D5D0C1;
        border-radius: 4px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.2rem;
    }
    .tech-panel-sky {
        background: #EBF4FA;
        border: 1px solid #B8D4E8;
        border-radius: 4px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1.25rem;
    }
    .tech-panel-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: #1E5D88 !important;
        margin-bottom: 0.4rem;
    }
    .tech-panel h4 {
        margin: 0 0 0.5rem 0;
        color: #0F293D !important;
        font-size: 1rem;
        font-weight: 700;
    }
    .tech-panel p, .tech-panel-sky p {
        margin: 0;
        color: #2F4252 !important;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .station-box {
        background-color: #FCFBF7;
        border: 1px solid #C9C3B2;
        border-radius: 4px;
        padding: 0.75rem 0.9rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        line-height: 1.6;
        margin-top: 0.5rem;
    }
    h2, h3, h4 {
        letter-spacing: -0.015em;
        color: #0F293D !important;
    }
"""

css_open = """
    /* Sidebar open state */
    section[data-testid="stMain"] {
        margin-left: 21rem !important; 
    }
</style>
"""

css_closed = """
    /* Sidebar closed state */
    section[data-testid="stSidebar"] { 
        display: none !important; 
    }
    section[data-testid="stMain"] {
        margin-left: 0px !important;
        padding-left: 0px !important;
        width: 100% !important;
    }
    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
</style>
"""

if st.session_state.sidebar_open:
    st.markdown(css_base + css_open, unsafe_allow_html=True)
else:
    st.markdown(css_base + css_closed, unsafe_allow_html=True)


LOCATION_CATALOG = [
    {"city": "Banda Aceh", "province": "Aceh", "island": "Sumatra", "base_temp": 27.1, "rad": 19.2, "precip": 5.4, "phours": 6.5, "ws": 12.5, "wg": 28.0, "wcode": 51.0},
    {"city": "Medan", "province": "Sumatera Utara", "island": "Sumatra", "base_temp": 26.8, "rad": 18.1, "precip": 7.8, "phours": 9.2, "ws": 10.4, "wg": 26.5, "wcode": 61.0},
    {"city": "Padang", "province": "Sumatera Barat", "island": "Sumatra", "base_temp": 25.4, "rad": 17.5, "precip": 11.2, "phours": 11.5, "ws": 9.8, "wg": 25.0, "wcode": 63.0},
    {"city": "Pekanbaru", "province": "Riau", "island": "Sumatra", "base_temp": 26.7, "rad": 18.0, "precip": 7.9, "phours": 8.8, "ws": 9.5, "wg": 25.5, "wcode": 61.0},
    {"city": "Batam", "province": "Kepulauan Riau", "island": "Sumatra", "base_temp": 27.0, "rad": 18.4, "precip": 6.8, "phours": 8.1, "ws": 13.2, "wg": 29.5, "wcode": 55.0},
    {"city": "Jambi", "province": "Jambi", "island": "Sumatra", "base_temp": 26.6, "rad": 17.9, "precip": 7.1, "phours": 8.5, "ws": 9.6, "wg": 24.8, "wcode": 61.0},
    {"city": "Palembang", "province": "Sumatera Selatan", "island": "Sumatra", "base_temp": 26.9, "rad": 18.3, "precip": 7.4, "phours": 8.6, "ws": 10.8, "wg": 26.2, "wcode": 61.0},
    {"city": "Bandar Lampung", "province": "Lampung", "island": "Sumatra", "base_temp": 26.2, "rad": 18.2, "precip": 6.5, "phours": 8.0, "ws": 10.9, "wg": 26.8, "wcode": 55.0},
    {"city": "Serang", "province": "Banten", "island": "Java", "base_temp": 26.7, "rad": 18.8, "precip": 5.9, "phours": 7.2, "ws": 11.8, "wg": 28.4, "wcode": 53.0},
    {"city": "Jakarta", "province": "DKI Jakarta", "island": "Java", "base_temp": 27.2, "rad": 18.9, "precip": 6.1, "phours": 7.0, "ws": 11.5, "wg": 27.9, "wcode": 53.0},
    {"city": "Bandung", "province": "Jawa Barat", "island": "Java", "base_temp": 23.0, "rad": 17.8, "precip": 7.5, "phours": 9.4, "ws": 10.2, "wg": 26.0, "wcode": 61.0},
    {"city": "Semarang", "province": "Jawa Tengah", "island": "Java", "base_temp": 26.8, "rad": 19.3, "precip": 6.4, "phours": 7.5, "ws": 12.1, "wg": 28.9, "wcode": 55.0},
    {"city": "Yogyakarta", "province": "DI Yogyakarta", "island": "Java", "base_temp": 25.3, "rad": 18.9, "precip": 6.2, "phours": 7.4, "ws": 11.9, "wg": 28.1, "wcode": 55.0},
    {"city": "Surabaya", "province": "Jawa Timur", "island": "Java", "base_temp": 27.4, "rad": 20.1, "precip": 5.3, "phours": 6.3, "ws": 13.0, "wg": 30.2, "wcode": 51.0},
    {"city": "Malang", "province": "Jawa Timur", "island": "Java", "base_temp": 24.1, "rad": 18.7, "precip": 6.8, "phours": 8.2, "ws": 10.6, "wg": 26.4, "wcode": 55.0},
    {"city": "Denpasar", "province": "Bali", "island": "Bali & Nusa Tenggara", "base_temp": 26.3, "rad": 20.4, "precip": 5.5, "phours": 6.2, "ws": 14.5, "wg": 32.1, "wcode": 51.0},
    {"city": "Mataram", "province": "Nusa Tenggara Barat", "island": "Bali & Nusa Tenggara", "base_temp": 26.1, "rad": 20.2, "precip": 5.2, "phours": 6.0, "ws": 13.4, "wg": 30.5, "wcode": 51.0},
    {"city": "Kupang", "province": "Nusa Tenggara Timur", "island": "Bali & Nusa Tenggara", "base_temp": 26.8, "rad": 21.5, "precip": 4.2, "phours": 4.8, "ws": 16.2, "wg": 35.8, "wcode": 51.0},
    {"city": "Pontianak", "province": "Kalimantan Barat", "island": "Kalimantan", "base_temp": 26.5, "rad": 18.1, "precip": 9.8, "phours": 10.6, "ws": 9.9, "wg": 25.7, "wcode": 61.0},
    {"city": "Palangka Raya", "province": "Kalimantan Tengah", "island": "Kalimantan", "base_temp": 26.7, "rad": 18.0, "precip": 8.6, "phours": 9.8, "ws": 9.4, "wg": 25.1, "wcode": 61.0},
    {"city": "Banjarmasin", "province": "Kalimantan Selatan", "island": "Kalimantan", "base_temp": 26.6, "rad": 18.2, "precip": 7.9, "phours": 9.1, "ws": 10.5, "wg": 26.3, "wcode": 61.0},
    {"city": "Balikpapan", "province": "Kalimantan Timur", "island": "Kalimantan", "base_temp": 26.5, "rad": 17.8, "precip": 8.1, "phours": 9.5, "ws": 11.2, "wg": 27.0, "wcode": 61.0},
    {"city": "Samarinda", "province": "Kalimantan Timur", "island": "Kalimantan", "base_temp": 26.4, "rad": 17.6, "precip": 8.4, "phours": 9.9, "ws": 9.7, "wg": 25.2, "wcode": 61.0},
    {"city": "Manado", "province": "Sulawesi Utara", "island": "Sulawesi", "base_temp": 25.9, "rad": 18.5, "precip": 8.3, "phours": 9.6, "ws": 11.4, "wg": 27.8, "wcode": 61.0},
    {"city": "Palu", "province": "Sulawesi Tengah", "island": "Sulawesi", "base_temp": 25.6, "rad": 18.3, "precip": 6.9, "phours": 8.4, "ws": 10.1, "wg": 25.9, "wcode": 55.0},
    {"city": "Makassar", "province": "Sulawesi Selatan", "island": "Sulawesi", "base_temp": 26.3, "rad": 19.5, "precip": 7.6, "phours": 7.9, "ws": 12.8, "wg": 29.8, "wcode": 55.0},
    {"city": "Kendari", "province": "Sulawesi Tenggara", "island": "Sulawesi", "base_temp": 26.4, "rad": 18.7, "precip": 7.2, "phours": 8.5, "ws": 11.6, "wg": 27.5, "wcode": 55.0},
    {"city": "Ambon", "province": "Maluku", "island": "Maluku", "base_temp": 26.5, "rad": 19.0, "precip": 8.9, "phours": 9.7, "ws": 13.5, "wg": 30.1, "wcode": 61.0},
    {"city": "Ternate", "province": "Maluku Utara", "island": "Maluku", "base_temp": 26.7, "rad": 19.4, "precip": 7.5, "phours": 8.7, "ws": 12.4, "wg": 28.6, "wcode": 55.0},
    {"city": "Sorong", "province": "Papua Barat Daya", "island": "Papua", "base_temp": 26.6, "rad": 18.6, "precip": 9.4, "phours": 10.3, "ws": 11.9, "wg": 28.2, "wcode": 61.0},
    {"city": "Jayapura", "province": "Papua", "island": "Papua", "base_temp": 26.2, "rad": 18.1, "precip": 8.2, "phours": 9.5, "ws": 10.7, "wg": 26.6, "wcode": 61.0},
]

CATALOG_DF = pd.DataFrame(LOCATION_CATALOG)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

@st.cache_resource(show_spinner="Initializing CatBoost model & spatial encoder...")
def load_ml_assets():
    model_candidates = [
        os.path.join(BASE_DIR, "catboost_temperature_model.cbm"),
        "catboost_temperature_model.cbm",
        r"G:\Data Analyst\Indonesia Weather\catboost_temperature_model.cbm"
    ]
    encoder_candidates = [
        os.path.join(BASE_DIR, "location_encoder.pkl"),
        "location_encoder.pkl",
        r"G:\Data Analyst\Indonesia Weather\location_encoder.pkl"
    ]

    loaded_model = None
    model_status = "SYNTHETIC CLIMATOLOGY (Missing .cbm)"
    for m_path in model_candidates:
        if os.path.exists(m_path):
            try:
                cb = CatBoostRegressor()
                with open(m_path, "rb") as f:
                    cb.load_model(blob=f.read())
                loaded_model = cb
                model_status = f"CATBOOST ACTIVE ({os.path.basename(m_path)})"
                break
            except Exception:
                try:
                    cb = CatBoostRegressor()
                    cb.load_model(m_path)
                    loaded_model = cb
                    model_status = f"CATBOOST ACTIVE ({os.path.basename(m_path)})"
                    break
                except Exception:
                    pass

    loaded_encoder = None
    for e_path in encoder_candidates:
        if os.path.exists(e_path):
            try:
                with open(e_path, "rb") as f:
                    loaded_encoder = joblib.load(f)
                break
            except Exception:
                pass

    return loaded_model, loaded_encoder, model_status

@st.cache_data(show_spinner="Loading historical climatology baselines...")
def load_historical_proxy():
    csv_candidates = [
        os.path.join(BASE_DIR, "idn_weather_daily.csv.gz"),
        os.path.join(BASE_DIR, "idn_weather_daily.csv"),
        "idn_weather_daily.csv.gz",
        "idn_weather_daily.csv",
        r"G:\Data Analyst\Indonesia Weather\idn_weather_daily.csv.gz",
        r"G:\Data Analyst\Indonesia Weather\idn_weather_daily.csv"
    ]
    proxy_cols = [
        "weather_code", "precipitation_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum"
    ]

    for c_path in csv_candidates:
        if os.path.exists(c_path):
            try:
                df = pd.read_csv(c_path)
                df["date"] = pd.to_datetime(df["date"])
                df["month"] = df["date"].dt.month
                df["day_of_year"] = df["date"].dt.dayofyear
                grouped = (
                    df.groupby(["city", "province", "island", "month", "day_of_year"])[proxy_cols]
                    .mean()
                    .reset_index()
                )
                grouped["weather_code"] = grouped["weather_code"].round(0)
                return grouped, True
            except Exception:
                pass

    rows = []
    for _, loc in CATALOG_DF.iterrows():
        is_south = loc["island"] in ["Java", "Bali & Nusa Tenggara", "Maluku", "Papua"]
        for doy in range(1, 367):
            m = min(12, (doy - 1) // 30 + 1)
            angle = 2 * np.pi * (doy - 200) / 365.25
            dry_factor = np.cos(angle) if is_south else 0.5 * np.cos(angle)
            rad = np.clip(loc["rad"] + 1.8 * dry_factor, 10.0, 27.5)
            precip = np.clip(loc["precip"] - 3.2 * dry_factor, 0.2, 25.0)
            phours = np.clip(loc["phours"] - 2.8 * dry_factor, 0.5, 20.0)
            ws = np.clip(loc["ws"] + 1.5 * dry_factor, 5.0, 28.0)
            wg = np.clip(loc["wg"] + 2.5 * dry_factor, 14.0, 50.0)
            wcode = 51.0 if precip < 5.0 else (55.0 if precip < 8.0 else 61.0)
            rows.append({
                "city": loc["city"],
                "province": loc["province"],
                "island": loc["island"],
                "month": m,
                "day_of_year": doy,
                "weather_code": wcode,
                "precipitation_sum": round(float(precip), 2),
                "precipitation_hours": round(float(phours), 1),
                "wind_speed_10m_max": round(float(ws), 2),
                "wind_gusts_10m_max": round(float(wg), 2),
                "shortwave_radiation_sum": round(float(rad), 2)
            })
    return pd.DataFrame(rows), False

model, encoder, model_status_text = load_ml_assets()
proxy_df, has_raw_csv = load_historical_proxy()

CITIES_SORTED = sorted(CATALOG_DF["city"].unique().tolist())
PROVINCES_SORTED = sorted(CATALOG_DF["province"].unique().tolist())
ISLANDS_SORTED = sorted(CATALOG_DF["island"].unique().tolist())

def encode_locations(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    cat_cols = ["city", "province", "island"]
    if encoder is not None:
        df_out[cat_cols] = encoder.transform(df_out[cat_cols])
    else:
        city_map = {c: float(i) for i, c in enumerate(CITIES_SORTED)}
        prov_map = {p: float(i) for i, p in enumerate(PROVINCES_SORTED)}
        isl_map = {isl: float(i) for i, isl in enumerate(ISLANDS_SORTED)}
        df_out["city"] = df_out["city"].map(city_map).fillna(-1.0)
        df_out["province"] = df_out["province"].map(prov_map).fillna(-1.0)
        df_out["island"] = df_out["island"].map(isl_map).fillna(-1.0)
    return df_out

def predict_dates_all_locations(
    date_list: list,
    custom_override: dict = None,
    target_city: str = None,
    scenario: str = "Neutral (Baseline)"
) -> pd.DataFrame:
    dt_series = pd.to_datetime(date_list)
    date_records = pd.DataFrame({
        "date": dt_series,
        "year": dt_series.year,
        "month": dt_series.month,
        "day_of_year": dt_series.dayofyear,
        "t_days": (dt_series - pd.Timestamp("2000-01-01")).days
    })

    merged = pd.merge(date_records, proxy_df, on=["month", "day_of_year"], how="left")

    if merged["shortwave_radiation_sum"].isnull().any():
        merged = merged.sort_values(["city", "day_of_year"])
        cols_to_fill = ["weather_code", "precipitation_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum"]
        merged[cols_to_fill] = merged.groupby("city")[cols_to_fill].ffill().bfill()

    if custom_override is not None and target_city is not None:
        mask = merged["city"] == target_city
        for k, v in custom_override.items():
            if k in merged.columns:
                merged.loc[mask, k] = v

    scenario_temp_offset = 0.0
    if scenario == "El Niño (Hot & Dry)":
        merged["precipitation_sum"] *= 0.60
        merged["precipitation_hours"] *= 0.60
        merged["shortwave_radiation_sum"] *= 1.15
        scenario_temp_offset = 0.8
    elif scenario == "La Niña (Cool & Wet)":
        merged["precipitation_sum"] *= 1.30
        merged["precipitation_hours"] *= 1.25
        merged["shortwave_radiation_sum"] *= 0.85
        scenario_temp_offset = -0.5

    merged["entity_label"] = merged["city"] + " (" + merged["province"] + ")"

    X_features = encode_locations(merged)
    default_feature_order = [
        "city", "province", "island", "weather_code",
        "precipitation_sum", "precipitation_hours",
        "wind_speed_10m_max", "wind_gusts_10m_max",
        "shortwave_radiation_sum", "year", "month", "day_of_year", "t_days"
    ]

    if model is not None:
        feat_names = model.feature_names_ if hasattr(model, "feature_names_") and model.feature_names_ else default_feature_order
        X_Ordered = X_features[feat_names]
        merged["predicted_temp_mean"] = model.predict(X_Ordered)
    else:
        base_map = CATALOG_DF.set_index("city")["base_temp"].to_dict()
        base_t = merged["city"].map(base_map).fillna(26.1)
        rad_effect = (merged["shortwave_radiation_sum"] - 18.3) * 0.22
        rain_effect = -(merged["precipitation_hours"] - 8.5) * 0.06
        seasonal = 0.45 * np.sin(2 * np.pi * (merged["day_of_year"] - 80) / 365.25)
        merged["predicted_temp_mean"] = base_t + rad_effect + rain_effect + seasonal

    years_since_2000 = merged["year"] - 2000
    trend_adjustment = years_since_2000 * 0.03
    
    internal_variability = (
        np.sin(merged["year"] * 1.5 + merged["day_of_year"] * 0.1) * 0.15
    ) + (
        np.cos(merged["year"] * 0.8 + merged["month"] * 1.2) * 0.10
    )
    
    merged["predicted_temp_mean"] += trend_adjustment + scenario_temp_offset + internal_variability

    return merged

def render_signature_linechart(
    df_pred: pd.DataFrame,
    time_col: str,
    x_labels: list,
    title_name: str,
    x_axis_title: str,
    selected_city: str,
    selected_province: str,
    grouping_mode: str = "City per Province — City (Province)",
    show_annotations: bool = True,
    x_rotation: int = None
):
    if grouping_mode.startswith("City"):
        group_col = "entity_label"
        user_target = f"{selected_city} ({selected_province})"
    else:
        group_col = "province"
        user_target = selected_province

    period_means_all = df_pred.groupby([group_col, time_col])["predicted_temp_mean"].mean().unstack()
    period_means_all = period_means_all[sorted(period_means_all.columns)]

    entity_means = period_means_all.mean(axis=1).sort_values(ascending=False)
    national_mean = df_pred["predicted_temp_mean"].mean()
    period_national = df_pred.groupby(time_col)["predicted_temp_mean"].mean().sort_index()

    top_2 = entity_means.index[:2].tolist()
    bottom_2 = entity_means.index[-2:].tolist()
    median_entity = entity_means.index[len(entity_means) // 2]

    base_selected = top_2 + bottom_2 + [median_entity]
    base_colors = ["#B8332A", "#D97724", "#165682", "#5DA2D5", "#7C8B99"]
    entity_colors = dict(zip(base_selected, base_colors))

    user_in_benchmark = user_target in base_selected
    selected_entities = list(base_selected)

    if not user_in_benchmark and user_target in entity_means.index:
        selected_entities.append(user_target)
        entity_colors[user_target] = "#2B7A57"

    national_color = "#182633"
    labels = {}
    total_entities = len(entity_means)

    for ent in selected_entities:
        avg_temp = entity_means[ent]
        diff_avg = avg_temp - national_mean
        diff_str = f"+{diff_avg:.1f}°C" if diff_avg > 0 else f"{diff_avg:.1f}°C"
        rank_num = int(entity_means.index.get_loc(ent)) + 1

        if ent in top_2:
            prefix = f"[MAX #{top_2.index(ent) + 1}]"
        elif ent in bottom_2:
            prefix = f"[MIN #{bottom_2.index(ent) + 1}]"
        elif ent == median_entity:
            prefix = "[MEDIAN]"
        else:
            prefix = f"[SELECTED • #{rank_num}/{total_entities}]"

        user_tag = " (Active)" if (ent == user_target and user_in_benchmark) else ""
        labels[ent] = f"{prefix} {ent}{user_tag} — {avg_temp:.1f}°C ({diff_str})"

    plot_data = period_means_all.loc[selected_entities].T
    x_indices = list(range(1, len(x_labels) + 1))
    plot_data.index = x_indices
    period_national.index = x_indices

    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=130)

    for ent in selected_entities:
        is_user_line = (ent == user_target)
        lw = 2.8 if is_user_line else 2.0
        ms = 6.5 if is_user_line else 5.0
        z_ord = 6 if is_user_line else 4
        ax.plot(
            plot_data.index,
            plot_data[ent],
            marker="o",
            markersize=ms,
            color=entity_colors[ent],
            label=labels[ent],
            linewidth=lw,
            zorder=z_ord
        )

    ax.plot(
        period_national.index,
        period_national.values,
        color=national_color,
        linestyle="--",
        linewidth=1.6,
        alpha=0.8,
        label=f"[NATIONAL MEAN] — {national_mean:.1f}°C",
        zorder=3
    )

    if show_annotations:
        n_lines = len(selected_entities) + 1
        if n_lines <= 6:
            y_offsets = [34, 17, 0, -17, -34, -51]
        else:
            y_offsets = [42, 26, 10, -6, -22, -38, -54]

        step = 1 if len(x_indices) <= 16 else (2 if len(x_indices) <= 24 else 3)

        for x_idx in x_indices[::step]:
            tick_points = []
            nat_val = period_national.loc[x_idx]

            for ent in selected_entities:
                val = plot_data.loc[x_idx, ent]
                diff = val - nat_val
                text = f"{val:.1f}°C ({diff:+.1f})"
                tick_points.append({"val": val, "text": text, "color": entity_colors[ent]})

            tick_points.append({"val": nat_val, "text": f"{nat_val:.1f}°C", "color": national_color})
            tick_points.sort(key=lambda item: item["val"], reverse=True)

            for i, point in enumerate(tick_points):
                text_color = "#FCFBF7" if point["color"] in ["#B8332A", "#165682", "#182633", "#2B7A57"] else "#182633"
                offset_y = y_offsets[i] if i < len(y_offsets) else -62

                ax.annotate(
                    point["text"],
                    (x_idx, point["val"]),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="600",
                    color=text_color,
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        fc=point["color"],
                        ec="none",
                        alpha=0.94
                    ),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=point["color"],
                        linewidth=0.9,
                        alpha=0.7,
                        shrinkA=0,
                        shrinkB=0
                    ),
                    zorder=7
                )

    ax.set_xticks(x_indices)
    
    if x_rotation is not None:
        rotation_deg = x_rotation
        ha_align = "center" if rotation_deg == 90 else "right"
    else:
        rotation_deg = 0 if len(x_labels) <= 12 else 30
        ha_align = "right" if rotation_deg > 0 else "center"
        
    ax.set_xticklabels(x_labels, rotation=rotation_deg, ha=ha_align, fontsize=9.5)
    
    ax.set_xlabel(x_axis_title, fontsize=10.5, fontweight="700", labelpad=10)
    ax.set_ylabel("Mean Surface Temperature (°C)", fontsize=10.5, fontweight="700", labelpad=10)
    ax.set_title(title_name, fontsize=12.5, fontweight="700", loc="left", pad=16)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - 1.5, y_max + 2.0)

    legend_title = "STATION BENCHMARKS" if grouping_mode.startswith("City") else "PROVINCIAL BENCHMARKS"
    leg = ax.legend(
        title=legend_title,
        title_fontsize=8.5,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=8.8,
        frameon=True,
        facecolor="#F7F4EB",
        edgecolor="#D5D0C1"
    )
    leg.get_title().set_fontweight("bold")
    ax.grid(alpha=0.45, linestyle="--", linewidth=0.7)
    plt.tight_layout()

    user_rank = int(entity_means.index.get_loc(user_target)) + 1 if user_target in entity_means.index else None
    user_avg = float(entity_means[user_target]) if user_target in entity_means.index else None

    if user_target in top_2:
        user_role = f"Upper Extreme (#{top_2.index(user_target) + 1} Warmest)"
    elif user_target in bottom_2:
        user_role = f"Lower Extreme (#{bottom_2.index(user_target) + 1} Coolest)"
    elif user_target == median_entity:
        user_role = "National Median Baseline"
    else:
        user_role = f"Rank #{user_rank} of {total_entities}"

    summary_info = {
        "user_target": user_target,
        "user_role": user_role,
        "user_avg": user_avg,
        "national_mean": float(national_mean),
        "top_1_hottest": top_2[0],
        "top_1_hottest_val": float(entity_means[top_2[0]]),
        "top_1_coldest": bottom_2[-1],
        "top_1_coldest_val": float(entity_means[bottom_2[-1]]),
        "median_entity": median_entity,
        "median_val": float(entity_means[median_entity]),
        "ranking_table": entity_means.reset_index().rename(
            columns={group_col: "Region", 0: "Average Temperature (°C)"}
        )
    }

    return fig, plot_data, summary_info

# Button to open sidebar (shows in main area if closed)
if not st.session_state.sidebar_open:
    if st.button("☰ Open Sidebar Menu", use_container_width=False):
        st.session_state.sidebar_open = True
        st.rerun()

with st.sidebar:
    # Button to close sidebar (shows inside sidebar)
    if st.button("✖ Close Sidebar Menu", use_container_width=True):
        st.session_state.sidebar_open = False
        st.rerun()

    # Logo image
    try:
        st.image("frieren logo.jfif", use_container_width=True)
    except Exception:
        st.markdown("<h2 style='text-align: center; color: #1E5D88;'>IDN WEATHER</h2>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title" style="margin-top: 1.5rem;">00 / WORKBENCH VIEW</div>', unsafe_allow_html=True)

    nav_menu = st.radio(
        "Navigation",
        options=[
            "Forecast Workbench",
            "Spatial Rankings",
            "Station Network",
            "Methodology & Specs"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section-title">01 / STATION SELECTION</div>', unsafe_allow_html=True)

    selected_province = st.selectbox(
        "Province",
        options=PROVINCES_SORTED,
        index=PROVINCES_SORTED.index("DKI Jakarta") if "DKI Jakarta" in PROVINCES_SORTED else 0
    )

    filtered_cities = sorted(
        CATALOG_DF[CATALOG_DF["province"] == selected_province]["city"].unique().tolist()
    )
    selected_city = st.selectbox(
        "City / Station",
        options=filtered_cities,
        index=0
    )

    selected_island = CATALOG_DF[
        (CATALOG_DF["province"] == selected_province) & (CATALOG_DF["city"] == selected_city)
    ]["island"].iloc[0]

    st.markdown(
        f"""
        <div class="station-box">
            <b>STATION:</b> {selected_city.upper()}<br>
            <b>PROVINCE:</b> {selected_province}<br>
            <b>REGION:</b> {selected_island} ({len(filtered_cities)} station)
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-section-title">02 / ATMOSPHERIC FORCING</div>', unsafe_allow_html=True)
    scenario_mode = st.selectbox(
        "ENSO Phase Regime",
        options=[
            "Neutral (Baseline)",
            "El Niño (Hot & Dry)",
            "La Niña (Cool & Wet)"
        ],
        index=0
    )

    st.markdown('<div class="sidebar-section-title">03 / TEMPORAL HORIZON</div>', unsafe_allow_html=True)
    prediction_mode = st.selectbox(
        "Aggregation Window",
        options=[
            "1. Single-Day Point Simulation",
            "2. 7-Day Synoptic Window",
            "3. Multi-Month Seasonal Profile"
        ],
        index=2
    )

    st.markdown('<div class="sidebar-section-title">04 / PLOT CONFIGURATION</div>', unsafe_allow_html=True)
    grouping_mode = st.selectbox(
        "Spatial Grouping",
        options=[
            "City per Province — City (Province)",
            "Province — Province"
        ],
        index=0
    )
    show_annotations = st.toggle("Display Data Callouts (°C)", value=True)

    st.markdown('<div class="sidebar-section-title">SYSTEM TELEMETRY</div>', unsafe_allow_html=True)
    st.caption(f"Engine: `{model_status_text}`")
    st.caption("Dataset: [Indonesia Daily Weather](https://www.kaggle.com/datasets/danielaijdkx/indonesia-daily-weather-31-cities) · [Open-Meteo ERA5](https://open-meteo.com) (CC BY 4.0)")

# Editorial Masthead
st.markdown(f"""
<div class="masthead">
    <div class="masthead-top">
        <span class="masthead-code">IDN-CLIM // 2M SURFACE TEMPERATURE FORECASTING SYSTEM</span>
        <span class="masthead-code">ERA5 REANALYSIS • 31 STATIONS • 2000–2035</span>
    </div>
    <h1 class="masthead-title">Indonesian Maritime Continent Temperature Workbench</h1>
    <p class="masthead-desc">
        Physics-informed gradient boosting projections (<code>temperature_2m_mean</code>) across 31 synoptic stations and 29 provinces. 
        Combines a tuned CatBoost baseline with thermodynamic ENSO forcing and decadal warming adjustments.
    </p>
</div>
""", unsafe_allow_html=True)

if nav_menu == "Forecast Workbench":
    
    with st.expander("Technical Note: Interpreting Multi-Year Curves & Hybrid ENSO Adjustments", expanded=False):
        st.markdown("""
        **Why consecutive years appear similar under identical scenarios:**
        1. **Climatological Forcing Inputs:** Tree-based models cannot extrapolate beyond historical bounds. Baseline daily meteorological drivers (solar radiation, precipitation, wind) are drawn from 26-year historical day-of-year climatology.
        2. **Decadal Warming Rate (+0.03 °C/year):** The linear surface warming trend adds +0.12 °C over a 4-year interval—physically realistic per BMKG records, but subtle on a 20°C–32°C axis.
        3. **Evaluating Physical Sensitivity:** To observe substantial thermodynamic shifts, compare **Neutral vs. El Niño (+0.8 °C offset, -40% rain, +15% insolation)** or **La Niña (-0.5 °C offset)** within the same calendar year.
        """)
    
    if prediction_mode.startswith("1."):
        st.markdown(f"### Single-Day Point Simulation — {selected_city}, {selected_province}")
        
        if scenario_mode != "Neutral (Baseline)":
            st.markdown(f"""
            <div class="tech-panel-sky">
                <div class="tech-panel-header">ACTIVE TELECONNECTIONS OVERRIDE // {scenario_mode.upper()}</div>
                <p>Climatological inputs are perturbed to reflect ENSO phase anomalies prior to CatBoost inference.</p>
            </div>
            """, unsafe_allow_html=True)

        col_cfg, col_sim = st.columns([1, 1.8])
        with col_cfg:
            target_date = st.date_input(
                "Target Observation Date",
                value=datetime.date(2027, 2, 15),
                min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date(2035, 12, 31)
            )
            use_custom_weather = st.checkbox(
                "Override Meteorological Inputs (Manual Sensitivity Test)",
                value=False
            )

        doy_val = pd.Timestamp(target_date).dayofyear
        city_proxy = proxy_df[
            (proxy_df["city"] == selected_city) & (proxy_df["day_of_year"] == doy_val)
        ]
        if city_proxy.empty:
            city_proxy = proxy_df[proxy_df["city"] == selected_city].iloc[[0]]
        row_p = city_proxy.iloc[0]

        custom_override = None
        with col_sim:
            if use_custom_weather:
                st.markdown("**Manual Meteorological Forcing Parameters**")
                s1, s2, s3 = st.columns(3)
                with s1:
                    sim_rad = st.slider("Shortwave Radiation (MJ/m²)", 1.0, 30.0, float(row_p["shortwave_radiation_sum"]), 0.1)
                    sim_wcode = st.selectbox("WMO Weather Code", [0.0, 1.0, 2.0, 3.0, 51.0, 53.0, 55.0, 61.0, 63.0, 65.0], index=4)
                with s2:
                    sim_psum = st.slider("Precipitation Sum (mm)", 0.0, 150.0, float(row_p["precipitation_sum"]), 0.5)
                    sim_phours = st.slider("Precipitation Hours (h)", 0.0, 24.0, float(row_p["precipitation_hours"]), 1.0)
                with s3:
                    sim_ws = st.slider("Max Wind 10m (km/h)", 2.0, 50.0, float(row_p["wind_speed_10m_max"]), 0.5)
                    sim_wg = st.slider("Max Wind Gusts (km/h)", 10.0, 90.0, float(row_p["wind_gusts_10m_max"]), 0.5)

                custom_override = {
                    "shortwave_radiation_sum": sim_rad,
                    "weather_code": sim_wcode,
                    "precipitation_sum": sim_psum,
                    "precipitation_hours": sim_phours,
                    "wind_speed_10m_max": sim_ws,
                    "wind_gusts_10m_max": sim_wg
                }
            else:
                st.markdown(f"""
                <div class="tech-panel">
                    <div class="tech-panel-header">CLIMATOLOGICAL BASELINE INPUTS // DOY {doy_val:03d}</div>
                    <p>
                        <b>{selected_city} ({selected_province})</b> on <b>{target_date.strftime("%d %b %Y")}</b>:<br>
                        Shortwave Radiation: <span class="mono-val">{row_p["shortwave_radiation_sum"]:.1f} MJ/m²</span> &nbsp;|&nbsp; 
                        Precipitation: <span class="mono-val">{row_p["precipitation_sum"]:.1f} mm ({row_p["precipitation_hours"]:.1f} h)</span> &nbsp;|&nbsp; 
                        Max Wind (10m): <span class="mono-val">{row_p["wind_speed_10m_max"]:.1f} km/h</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)

        day_df = predict_dates_all_locations(
            [target_date], 
            custom_override=custom_override, 
            target_city=selected_city,
            scenario=scenario_mode
        )
        day_sorted = day_df.sort_values("predicted_temp_mean", ascending=False).reset_index(drop=True)
        day_sorted["rank"] = day_sorted.index + 1

        user_row = day_sorted[day_sorted["city"] == selected_city].iloc[0]
        nat_avg = day_sorted["predicted_temp_mean"].mean()
        diff_nat = user_row["predicted_temp_mean"] - nat_avg

        top_2_cities = day_sorted.iloc[:2]["city"].tolist()
        bottom_2_cities = day_sorted.iloc[-2:]["city"].tolist()
        median_city = day_sorted.iloc[len(day_sorted) // 2]["city"]

        if selected_city in top_2_cities:
            status_badge = f"Upper Extreme (#{top_2_cities.index(selected_city) + 1} Warmest)"
        elif selected_city in bottom_2_cities:
            status_badge = f"Lower Extreme (#{bottom_2_cities.index(selected_city) + 1} Coolest)"
        elif selected_city == median_city:
            status_badge = "National Median"
        else:
            status_badge = f"Rank #{int(user_row['rank'])} of {len(day_sorted)} Stations"

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            delta_cls = "kpi-delta-pos" if diff_nat >= 0 else "kpi-delta-neg"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">PROJECTED MEAN ({selected_city})</div>
                <div class="kpi-value">{user_row['predicted_temp_mean']:.2f}°C</div>
                <span class="{delta_cls}">{diff_nat:+.2f}°C vs National Mean</span>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">NATIONAL DISTRIBUTION PERCENTILE</div>
                <div class="kpi-value" style="font-size:1.15rem; padding-top:0.35rem;">{status_badge}</div>
                <span class="kpi-delta-neutral">Province: {selected_province}</span>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            hottest_r = day_sorted.iloc[0]
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">WARMEST STATION (MAX #1)</div>
                <div class="kpi-value">{hottest_r['predicted_temp_mean']:.2f}°C</div>
                <span class="kpi-delta-pos">{hottest_r['city']} ({hottest_r['province']})</span>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            coldest_r = day_sorted.iloc[-1]
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">COOLEST STATION (MIN #1)</div>
                <div class="kpi-value">{coldest_r['predicted_temp_mean']:.2f}°C</div>
                <span class="kpi-delta-neg">{coldest_r['city']} ({coldest_r['province']})</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### Cross-Station Distribution ({target_date.strftime('%d %b %Y')})")

        bench_Indices = [0, 1, len(day_sorted) // 2, len(day_sorted) - 2, len(day_sorted) - 1]
        bench_df = day_sorted.iloc[bench_Indices].copy()
        if selected_city not in bench_df["city"].values:
            bench_df = pd.concat([bench_df, day_sorted[day_sorted["city"] == selected_city]], ignore_index=True)
        bench_df = bench_df.sort_values("predicted_temp_mean", ascending=True)

        fig_bar, ax_bar = plt.subplots(figsize=(12, 4.6), dpi=120)
        bar_colors = []
        bar_labels = []
        for _, r in bench_df.iterrows():
            c_name = r["city"]
            p_name = r["province"]
            tag = " [Active]" if c_name == selected_city else ""
            if c_name == top_2_cities[0]:
                bar_colors.append("#B8332A")
                bar_labels.append(f"[MAX #1] {c_name} ({p_name}){tag}")
            elif c_name == top_2_cities[1]:
                bar_colors.append("#D97724")
                bar_labels.append(f"[MAX #2] {c_name} ({p_name}){tag}")
            elif c_name == bottom_2_cities[1]:
                bar_colors.append("#165682")
                bar_labels.append(f"[MIN #1] {c_name} ({p_name}){tag}")
            elif c_name == bottom_2_cities[0]:
                bar_colors.append("#5DA2D5")
                bar_labels.append(f"[MIN #2] {c_name} ({p_name}){tag}")
            elif c_name == median_city:
                bar_colors.append("#7C8B99")
                bar_labels.append(f"[MEDIAN] {c_name} ({p_name}){tag}")
            else:
                bar_colors.append("#2B7A57")
                bar_labels.append(f"[RANK #{int(r['rank'])}] {c_name} ({p_name})")

        bars = ax_bar.barh(bar_labels, bench_df["predicted_temp_mean"], color=bar_colors, edgecolor="none", height=0.55)
        ax_bar.axvline(nat_avg, color="#182633", linestyle="--", linewidth=1.5, label=f"National Mean ({nat_avg:.2f}°C)")
        ax_bar.set_xlim(bench_df["predicted_temp_mean"].min() - 1.5, bench_df["predicted_temp_mean"].max() + 1.8)
        ax_bar.set_xlabel("Projected 2m Mean Temperature (°C)", fontweight="bold")
        for b in bars:
            w = b.get_width()
            diff_b = w - nat_avg
            ax_bar.text(w + 0.08, b.get_y() + b.get_height() / 2, f"{w:.2f}°C ({diff_b:+.2f}°C)", va="center", fontweight="600", fontsize=8.8)
        ax_bar.legend(loc="lower right", frameon=True, facecolor="#F7F4EB", edgecolor="#D5D0C1")
        ax_bar.grid(axis="x", alpha=0.45, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    elif prediction_mode.startswith("2."):
        st.markdown(f"### 7-Day Synoptic Forecast — {selected_city} ({selected_province})")
        
        c_date1, c_date2 = st.columns([1, 2.2])
        with c_date1:
            start_date_w = st.date_input(
                "Synoptic Window Start Date",
                value=datetime.date(2027, 2, 1),
                min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date(2035, 12, 20)
            )
        end_date_w = start_date_w + datetime.timedelta(days=6)
        with c_date2:
            st.markdown(f"""
            <div class="tech-panel">
                <div class="tech-panel-header">HORIZON WINDOW // {start_date_w.strftime('%d %b %Y').upper()} – {end_date_w.strftime('%d %b %Y').upper()}</div>
                <p>
                    7-day trajectory comparing <b>{selected_city} ({selected_province})</b> against national upper/lower temperature extremes under <b>{scenario_mode}</b> conditions.
                </p>
            </div>
            """, unsafe_allow_html=True)

        date_list_w = [start_date_w + datetime.timedelta(days=i) for i in range(7)]
        df_week = predict_dates_all_locations(date_list_w, scenario=scenario_mode)

        date_to_idx = {pd.Timestamp(d): idx for idx, d in enumerate(date_list_w)}
        df_week["time_idx"] = df_week["date"].map(date_to_idx)
        x_labels_w = [d.strftime("%a, %d %b") for d in date_list_w]

        fig_w, plot_data_w, info_w = render_signature_linechart(
            df_pred=df_week,
            time_col="time_idx",
            x_labels=x_labels_w,
            title_name=f"7-Day Surface Temperature Trajectory ({start_date_w.strftime('%d %b')} – {end_date_w.strftime('%d %b %Y')})",
            x_axis_title="Synoptic Date",
            selected_city=selected_city,
            selected_province=selected_province,
            grouping_mode=grouping_mode,
            show_annotations=show_annotations,
            x_rotation=0
        )

        kw1, kw2, kw3, kw4 = st.columns(4)
        with kw1:
            diff_u = info_w["user_avg"] - info_w["national_mean"]
            cls_u = "kpi-delta-pos" if diff_u >= 0 else "kpi-delta-neg"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">7-DAY MEAN ({info_w['user_target']})</div>
                <div class="kpi-value">{info_w['user_avg']:.2f}°C</div>
                <span class="{cls_u}">{diff_u:+.2f}°C vs National ({info_w['national_mean']:.2f}°C)</span>
            </div>
            """, unsafe_allow_html=True)
        with kw2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">BENCHMARK STANDING</div>
                <div class="kpi-value" style="font-size:1.15rem; padding-top:0.35rem;">{info_w['user_role']}</div>
                <span class="kpi-delta-neutral">Scenario: {scenario_mode}</span>
            </div>
            """, unsafe_allow_html=True)
        with kw3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">UPPER EXTREME (MAX #1)</div>
                <div class="kpi-value">{info_w['top_1_hottest_val']:.2f}°C</div>
                <span class="kpi-delta-pos">{info_w['top_1_hottest']}</span>
            </div>
            """, unsafe_allow_html=True)
        with kw4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">LOWER EXTREME (MIN #1)</div>
                <div class="kpi-value">{info_w['top_1_coldest_val']:.2f}°C</div>
                <span class="kpi-delta-neg">{info_w['top_1_coldest']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig_w)
        plt.close(fig_w)

        with st.expander("View Numerical 7-Day Matrix"):
            disp_w = plot_data_w.copy()
            disp_w.index = x_labels_w
            st.dataframe(disp_w.style.format("{:.2f}°C"), use_container_width=True)

    elif prediction_mode.startswith("3."):
        st.markdown(f"### Multi-Month Seasonal Profile — {selected_city} ({selected_province})")

        month_names_list = list(calendar.month_name)[1:]
        c_mm1, c_mm2, c_mm3, c_mm4 = st.columns([1, 1.1, 1.1, 1.3])
        with c_mm1:
            sel_year_mm = st.number_input("Target Year", min_value=2000, max_value=2035, value=2027, step=1)
        with c_mm2:
            start_month_name = st.selectbox("Start Month", options=month_names_list, index=0)
            start_m_idx = month_names_list.index(start_month_name) + 1
        with c_mm3:
            valid_end_months = month_names_list[start_m_idx - 1:]
            default_end_idx = len(valid_end_months) - 1
            end_month_name = st.selectbox("End Month", options=valid_end_months, index=default_end_idx)
            end_m_idx = month_names_list.index(end_month_name) + 1
        with c_mm4:
            agg_style_mm = st.selectbox(
                "Temporal Resolution",
                options=[
                    "Monthly (Jan, Feb, Mar...)",
                    "Weekly (Month, Week 1..4)"
                ],
                index=0
            )

        start_dt_mm = datetime.date(int(sel_year_mm), int(start_m_idx), 1)
        last_day_end = calendar.monthrange(int(sel_year_mm), int(end_m_idx))[1]
        end_dt_mm = datetime.date(int(sel_year_mm), int(end_m_idx), last_day_end)

        total_days_mm = (end_dt_mm - start_dt_mm).days + 1
        date_list_mm = [start_dt_mm + datetime.timedelta(days=i) for i in range(total_days_mm)]
        df_multi = predict_dates_all_locations(date_list_mm, scenario=scenario_mode)

        if agg_style_mm.startswith("Weekly"):
            df_multi["week_in_month"] = df_multi["date"].dt.day.apply(lambda d: min(4, (d - 1) // 7 + 1))
            df_multi["period_order"] = df_multi["month"] * 10 + df_multi["week_in_month"]

            ordered_periods = sorted(df_multi["period_order"].unique().tolist())
            x_labels_mm = []
            for p_code in ordered_periods:
                m_num = p_code // 10
                w_num = p_code % 10
                m_str = calendar.month_abbr[m_num]
                x_labels_mm.append(f"{m_str} W{w_num}")

            time_col_mm = "period_order"
            x_title_mm = f"Weekly Aggregation ({start_month_name} – {end_month_name} {sel_year_mm})"
            rot_mm = 90
        else:
            time_col_mm = "month"
            ordered_months = sorted(df_multi["month"].unique().tolist())
            x_labels_mm = [calendar.month_abbr[m] for m in ordered_months]
            x_title_mm = f"Calendar Month ({start_month_name} – {end_month_name} {sel_year_mm})"
            rot_mm = 0

        fig_mm, plot_data_mm, info_mm = render_signature_linechart(
            df_pred=df_multi,
            time_col=time_col_mm,
            x_labels=x_labels_mm,
            title_name=f"Seasonal Temperature Regime ({start_month_name} – {end_month_name} {sel_year_mm}) // {scenario_mode}",
            x_axis_title=x_title_mm,
            selected_city=selected_city,
            selected_province=selected_province,
            grouping_mode=grouping_mode,
            show_annotations=show_annotations,
            x_rotation=rot_mm
        )

        kmm1, kmm2, kmm3, kmm4 = st.columns(4)
        with kmm1:
            diff_umm = info_mm["user_avg"] - info_mm["national_mean"]
            cls_umm = "kpi-delta-pos" if diff_umm >= 0 else "kpi-delta-neg"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">PERIOD MEAN ({info_mm['user_target']})</div>
                <div class="kpi-value">{info_mm['user_avg']:.2f}°C</div>
                <span class="{cls_umm}">{diff_umm:+.2f}°C vs National ({info_mm['national_mean']:.2f}°C)</span>
            </div>
            """, unsafe_allow_html=True)
        with kmm2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">BENCHMARK STANDING</div>
                <div class="kpi-value" style="font-size:1.15rem; padding-top:0.35rem;">{info_mm['user_role']}</div>
                <span class="kpi-delta-neutral">Scenario: {scenario_mode}</span>
            </div>
            """, unsafe_allow_html=True)
        with kmm3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">UPPER EXTREME (MAX #1)</div>
                <div class="kpi-value">{info_mm['top_1_hottest_val']:.2f}°C</div>
                <span class="kpi-delta-pos">{info_mm['top_1_hottest']}</span>
            </div>
            """, unsafe_allow_html=True)
        with kmm4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">LOWER EXTREME (MIN #1)</div>
                <div class="kpi-value">{info_mm['top_1_coldest_val']:.2f}°C</div>
                <span class="kpi-delta-neg">{info_mm['top_1_coldest']}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.pyplot(fig_mm)
        plt.close(fig_mm)

        with st.expander("Export Seasonal Forecast Matrix (CSV)"):
            disp_mm = plot_data_mm.copy()
            disp_mm.index = x_labels_mm
            st.dataframe(disp_mm.style.format("{:.2f}°C"), use_container_width=True)
            csv_bytes = disp_mm.to_csv().encode("utf-8")
            st.download_button(
                label="Download CSV Table",
                data=csv_bytes,
                file_name=f"idn_temp_forecast_{selected_city}_{sel_year_mm}.csv",
                mime="text/csv"
            )

elif nav_menu == "Spatial Rankings":
    st.markdown("### Spatial Temperature Rankings Across 31 Indonesian Stations")
    st.caption("Annualized distribution of projected 2m mean temperatures, shortwave insolation, and daily precipitation.")

    col_r1, col_r2 = st.columns([1, 2.5])
    with col_r1:
        rank_year = st.number_input("Evaluation Year", min_value=2000, max_value=2035, value=2027, step=1)
    with col_r2:
        sample_dates = [datetime.date(int(rank_year), m, 15) for m in range(1, 13)]
        df_rank_year = predict_dates_all_locations(sample_dates, scenario=scenario_mode)

    rank_summary = (
        df_rank_year.groupby(["city", "province", "island"])
        .agg(
            Mean_Temp=("predicted_temp_mean", "mean"),
            Max_Month_Temp=("predicted_temp_mean", "max"),
            Min_Month_Temp=("predicted_temp_mean", "min"),
            Avg_Solar_Rad=("shortwave_radiation_sum", "mean"),
            Avg_Precip=("precipitation_sum", "mean")
        )
        .reset_index()
        .sort_values("Mean_Temp", ascending=False)
        .reset_index(drop=True)
    )
    rank_summary["Rank"] = rank_summary.index + 1
    nat_mean_yr = rank_summary["Mean_Temp"].mean()
    rank_summary["Anomaly vs National (°C)"] = rank_summary["Mean_Temp"] - nat_mean_yr

    med_idx = len(rank_summary) // 2
    kategori_list = []
    for idx, r in rank_summary.iterrows():
        if idx == 0:
            kat = "MAX #1 (Warmest)"
        elif idx == 1:
            kat = "MAX #2 (Warmest)"
        elif idx == len(rank_summary) - 1:
            kat = "MIN #1 (Coolest)"
        elif idx == len(rank_summary) - 2:
            kat = "MIN #2 (Coolest)"
        elif idx == med_idx:
            kat = "MEDIAN BASELINE"
        else:
            kat = "Mid-Range"
        if r["city"] == selected_city:
            kat += " [Active]"
        kategori_list.append(kat)

    rank_summary["Classification"] = kategori_list

    st.dataframe(
        rank_summary[[
            "Rank", "Classification", "city", "province", "island",
            "Mean_Temp", "Anomaly vs National (°C)", "Min_Month_Temp", "Max_Month_Temp",
            "Avg_Solar_Rad", "Avg_Precip"
        ]].rename(columns={
            "city": "Station / City",
            "province": "Province",
            "island": "Archipelago",
            "Mean_Temp": "Annual Mean (°C)",
            "Min_Month_Temp": "Min Month (°C)",
            "Max_Month_Temp": "Max Month (°C)",
            "Avg_Solar_Rad": "Insolation (MJ/m²)",
            "Avg_Precip": "Precip (mm/d)"
        }).style.format({
            "Annual Mean (°C)": "{:.2f}°C",
            "Anomaly vs National (°C)": "{:+.2f}°C",
            "Min Month (°C)": "{:.2f}°C",
            "Max Month (°C)": "{:.2f}°C",
            "Insolation (MJ/m²)": "{:.2f}",
            "Precip (mm/d)": "{:.2f}"
        }),
        use_container_width=True,
        height=620
    )

elif nav_menu == "Station Network":
    st.markdown("### Synoptic Station Network & Provincial Coverage")
    st.caption("Inventory of the 31 representative meteorological stations across 29 Indonesian provinces included in the ERA5 training catalog.")
    
    coverage_df = (
        CATALOG_DF.groupby(["island", "province"])
        .agg(
            City_Count=("city", "nunique"),
            Cities=("city", lambda x: ", ".join(sorted(x)))
        )
        .reset_index()
        .sort_values(["island", "province"])
    )
    
    cov1, cov2, cov3, cov4 = st.columns(4)
    with cov1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">PROVINCIAL JURISDICTIONS</div>
            <div class="kpi-value">{coverage_df["province"].nunique()}</div>
            <span class="kpi-delta-neutral">Represented in Catalog</span>
        </div>
        """, unsafe_allow_html=True)
    with cov2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">SYNOPTIC STATIONS</div>
            <div class="kpi-value">{CATALOG_DF["city"].nunique()}</div>
            <span class="kpi-delta-neutral">Active City Nodes</span>
        </div>
        """, unsafe_allow_html=True)
    with cov3:
        multi_city = int((coverage_df["City_Count"] > 1).sum())
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">MULTI-STATION PROVINCES</div>
            <div class="kpi-value">{multi_city}</div>
            <span class="kpi-delta-neg">Jawa Timur, Kaltim</span>
        </div>
        """, unsafe_allow_html=True)
    with cov4:
        single_city = int((coverage_df["City_Count"] == 1).sum())
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">SINGLE-STATION PROVINCES</div>
            <div class="kpi-value">{single_city}</div>
            <span class="kpi-delta-neutral">1 Representative Node</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    search_query = st.text_input(
        "Filter by Province or Station Name",
        value="",
        placeholder="Type province or city name (e.g., Jawa, Papua, Semarang)..."
    ).strip().lower()
    
    display_df = coverage_df.copy()
    if search_query:
        display_df = display_df[
            display_df["province"].str.lower().str.contains(search_query, na=False)
            | display_df["Cities"].str.lower().str.contains(search_query, na=False)
        ]
    
    st.dataframe(
        display_df.rename(columns={
            "island": "Island Group",
            "province": "Province",
            "City_Count": "Station Count",
            "Cities": "Registered Stations"
        }),
        use_container_width=True,
        height=480
    )

elif nav_menu == "Methodology & Specs":
    st.markdown("### System Architecture & Physical Modeling Specifications")

    col_intro1, col_intro2 = st.columns([1.35, 1])
    with col_intro1:
        st.markdown("""
        <div class="tech-panel">
            <div class="tech-panel-header">01 // CHRONOLOGICAL SPLIT & LEAKAGE PREVENTION</div>
            <h4>Strict Temporal Holdout Validation</h4>
            <p>
                Trained on <b>303,645 daily observations</b> (2000–2026) partitioned chronologically: 
                <b>Training Set (2000–2021)</b>, <b>Validation Set (2022–2023)</b>, and <b>Unseen Test Set (2024–2026)</b>. 
                Variables with direct mathematical or physical circularity—specifically <code>temperature_2m_max</code>, 
                <code>temperature_2m_min</code>, and <code>et0_fao_evapotranspiration</code>—were excluded from the feature space 
                to force the model to learn true radiative and advective drivers.
            </p>
        </div>
        <div class="tech-panel">
            <div class="tech-panel-header">02 // HYBRID POST-PROCESSING</div>
            <h4>Physics-Informed Extrapolation & ENSO Perturbations</h4>
            <p>
                Because decision-tree ensembles cannot extrapolate linear trends beyond their training domain, 
                the baseline CatBoost prediction is coupled with three domain-driven adjustments:
                <br>• <b>Decadal Surface Warming (+0.03 °C/yr):</b> Aligned with BMKG 1981–2020 observational warming trends.
                <br>• <b>ENSO Thermodynamic Forcing:</b> El Niño applies -40% precipitation, +15% shortwave radiation, and a +0.8 °C anomaly offset; La Niña applies +30% precipitation, -15% radiation, and a -0.5 °C offset.
                <br>• <b>Harmonic Internal Variability:</b> Bounded sub-seasonal oscillation (±0.15 °C) to reflect non-stationary atmospheric variance.
            </p>
        </div>
        <div class="tech-panel">
            <div class="tech-panel-header">03 // DATA PROVENANCE</div>
            <h4>ERA5 Reanalysis & Open-Meteo Archive</h4>
            <p>
                Historical meteorological records sourced from 
                <a href="https://www.kaggle.com/datasets/danielaijdkx/indonesia-daily-weather-31-cities" target="_blank">Indonesia Daily Weather: 31 Cities since 2000</a> 
                (Daniel Silalahi) via <a href="https://open-meteo.com" target="_blank">Open-Meteo</a> (CC BY 4.0), incorporating modified Copernicus Climate Change Service (ERA5) reanalysis data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_intro2:
        st.markdown("#### Empirical Validation Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">VALIDATION SET (2022–2023)</div>
                <div class="kpi-value">0.42°C</div>
                <span class="kpi-delta-neg">MAE • RMSE: 0.54°C (R² = 0.836)</span>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown("""
            <div class="kpi-card">
                <div class="kpi-label">UNSEEN TEST SET (2024–2026)</div>
                <div class="kpi-value">0.60°C</div>
                <span class="kpi-delta-neutral">MAE • RMSE: 0.79°C (R² = 0.679)</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Production Hyperparameters (`CatBoostRegressor`)**")
        st.code(
            """CatBoostRegressor(
    learning_rate=0.05,
    depth=8,
    iterations=500,
    l2_leaf_reg=1,
    random_state=42
)""",
            language="python"
        )
