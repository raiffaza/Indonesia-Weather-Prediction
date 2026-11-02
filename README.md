# 🇮🇩 BMKG Climate AI — Indonesia Temperature Forecasting System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![CatBoost](https://img.shields.io/badge/ML-CatBoost-yellow.svg)
![Status](https://img.shields.io/badge/Status-Development-green.svg)

An interactive meteorological analytics platform designed to project daily mean temperatures (`temperature_2m_mean`) across 31 cities and 29 provinces in Indonesia. 

This project goes beyond standard Machine Learning by implementing a **Hybrid Physics-Informed Architecture**, overcoming the inherent limitation of tree-based models in forecasting long-term climate trends.

## 🚀 Key Features

*   **📊 3 Forecast Horizons:** 
    *   *Daily:* Single-day prediction with a What-If weather parameter simulator.
    *   *Weekly:* 7-day forecast line charts comparing against national extremes.
    *   *Multi-Month:* Cross-month trends aggregated weekly (e.g., *February, Week 1*).
*   **🌡️ Climate Scenario Modeling:** Simulate extreme climate anomalies dynamically. Switch between **Neutral**, **El Niño (Hot & Dry)**, and **La Niña (Cool & Wet)** scenarios to see the direct thermodynamic impact on temperature.
*   **🏆 Automated Benchmarking:** Instantly identifies the **Top 2 Hottest**, **Top 2 Coldest**, and **Median** regions for any given period.
*   **📍 Data Transparency:** A dedicated page showcasing exactly which cities are available in the dataset, preventing user confusion.

## 🧠 The Hybrid Physics-Informed Architecture

Tree-based models (like CatBoost) are excellent at learning seasonal patterns but **cannot extrapolate future trends** or understand phenomena like ENSO. To solve this, the final prediction is a combination of:

1.  **CatBoost Base Prediction:** Learns complex monsoon cycles and spatial interactions from 300,000+ historical daily weather observations (2000–2026).
2.  **Physics-Informed Post-Processing:** 
    *   **Global Warming Trend:** Adds `+0.03 °C / year` based on BMKG historical warming rates.
    *   **ENSO Thermodynamic Offset:** Adds `+0.8 °C` for El Niño, and `-0.5 °C` for La Niña to simulate realistic extreme temperatures.
    *   **Stochastic Noise:** Injects bounded random fluctuations (`±0.15 °C`) based on the year and month to simulate the chaotic nature of atmospheric variability (preventing flat-line multi-year projections).

## 🛠️ Tech Stack
*   **Language:** Python
*   **App Framework:** Streamlit
*   **Machine Learning:** CatBoost, Scikit-learn (for preprocessing)
*   **Data Processing:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn

## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/raiffaza/Indonesia-Weather-Prediction.git
   cd Indonesia-Weather-Prediction
