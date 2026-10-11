
"""
Melbourne Daily Minimum Temperature Forecasting App

Install dependencies:
    pip install streamlit pandas numpy matplotlib statsmodels

Run:
    streamlit run app.py
"""

from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Melbourne Temperature Forecast",
    page_icon="🌡️",
    layout="wide"
)

st.title("Melbourne Daily Minimum Temperature Forecast")

st.write(
    "Forecast future daily minimum temperatures using "
    "a pretrained Fourier + ARIMA model."
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "CP2_data_ts.csv"
MODEL_FILE = BASE_DIR / "fourier_arima_model.pkl"


# ==================================================
# 2. Fourier feature generation
# ==================================================

def make_fourier_features(
    dates,
    period=365.25,
    harmonics=2
):
    """
    Generate annual Fourier features using actual calendar dates.

    The feature order must match the training procedure:
    sin_1, cos_1, sin_2, cos_2.
    """

    dates = pd.DatetimeIndex(dates)

    elapsed_days = (
        dates - pd.Timestamp("1981-01-01")
    ).days.to_numpy()

    features = pd.DataFrame(index=dates)

    for k in range(1, harmonics + 1):
        features[f"sin_{k}"] = np.sin(
            2 * np.pi * k * elapsed_days / period
        )

        features[f"cos_{k}"] = np.cos(
            2 * np.pi * k * elapsed_days / period
        )

    return features


# ==================================================
# 3. Load and clean historical data
# ==================================================

@st.cache_data
def load_and_clean_data(file_path, modified_time):
    """
    Load the CSV, convert invalid temperatures to NaN,
    and interpolate missing values by time.
    """

    raw = pd.read_csv(file_path)

    if raw.shape[1] < 2:
        raise ValueError(
            "The CSV must contain date and temperature columns."
        )

    data = raw.iloc[:, :2].copy()

    data.columns = [
        "date_original",
        "temperature_original"
    ]

    # Parse dates in day-month-year format.
    data["date"] = pd.to_datetime(
        data["date_original"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Fallback for a different date format.
    if data["date"].isna().any():
        data["date"] = pd.to_datetime(
            data["date_original"],
            errors="coerce",
            dayfirst=True
        )

    # Convert temperatures to numeric.
    data["temperature"] = pd.to_numeric(
        data["temperature_original"].astype(str).str.strip(),
        errors="coerce"
    )

    # Remove invalid dates and sort chronologically.
    data = data.dropna(subset=["date"])
    data = data.sort_values("date")

    # Ensure one observation per date.
    data = data.drop_duplicates(
        subset=["date"],
        keep="first"
    )

    data = data.set_index("date")

    if data["temperature"].notna().sum() < 100:
        raise ValueError(
            "Too few valid temperature observations were found."
        )

    # Interpolate invalid temperatures using the time index.
    data["temperature"] = data["temperature"].interpolate(
        method="time"
    )

    data["temperature"] = (
        data["temperature"].ffill().bfill()
    )

    if data["temperature"].isna().any():
        raise ValueError(
            "Missing temperature values remain after cleaning."
        )

    return data[["temperature"]]


# ==================================================
# 4. Load the saved fitted model
# ==================================================

@st.cache_resource
def load_saved_model(model_path, modified_time):
    """
    Load the fitted Statsmodels result saved as a pickle file.

    Only load pickle files from a trusted source.
    """

    with open(model_path, "rb") as file:
        fitted_model = pickle.load(file)

    return fitted_model


# Check required files.
missing_files = []

if not DATA_FILE.exists():
    missing_files.append(DATA_FILE.name)

if not MODEL_FILE.exists():
    missing_files.append(MODEL_FILE.name)

if missing_files:
    st.error(
        "The following required files are missing: "
        + ", ".join(missing_files)
        + ". Place them in the same folder as app.py."
    )
    st.stop()


# Load historical data.
try:
    history = load_and_clean_data(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# Load saved model.
try:
    fitted_model = load_saved_model(
        str(MODEL_FILE),
        MODEL_FILE.stat().st_mtime
    )
except Exception as exc:
    st.error(
        f"Could not load the saved model: {exc}. "
        "Check that it was saved using a compatible "
        "Python and Statsmodels environment."
    )
    st.stop()


# ==================================================
# 5. Forecast horizon input
# ==================================================

st.sidebar.header("Forecast Settings")

months_ahead = st.sidebar.number_input(
    "How many future months?",
    min_value=1,
    max_value=36,
    value=12,
    step=1,
    help="Choose between 1 and 36 future months."
)

last_date = history.index.max()

forecast_start = last_date + pd.Timedelta(days=1)

forecast_end = last_date + pd.DateOffset(
    months=int(months_ahead)
)

future_dates = pd.date_range(
    start=forecast_start,
    end=forecast_end,
    freq="D"
)

if len(future_dates) == 0:
    st.error("No future dates were generated.")
    st.stop()


# ==================================================
# 6. Historical data overview
# ==================================================

st.subheader("Historical Dataset")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Historical observations",
    f"{len(history):,}"
)

col2.metric(
    "First observation",
    history.index.min().strftime("%d %b %Y")
)

col3.metric(
    "Last observation",
    last_date.strftime("%d %b %Y")
)

st.line_chart(
    history["temperature"],
    height=280
)


# ==================================================
# 7. Generate multi-step forecasts
# ==================================================

st.subheader("Temperature Forecast")

with st.spinner("Generating predictions using the saved model..."):

    try:
        # Generate future Fourier features using real dates.
        future_exog = make_fourier_features(
            future_dates,
            period=365.25,
            harmonics=2
        )

        # The saved model was fitted using an integer time index.
        future_exog.index = pd.RangeIndex(
            start=len(history),
            stop=len(history) + len(future_dates)
        )

        # Forecast all requested future days at once.
        forecast_values = np.asarray(
            fitted_model.forecast(
                steps=len(future_dates),
                exog=future_exog
            ),
            dtype=float
        )

        if len(forecast_values) != len(future_dates):
            raise ValueError(
                "The forecast length does not match the requested dates."
            )

    except Exception as exc:
        st.error(
            f"Forecasting failed: {exc}. "
            "Check that the saved model was fitted on the same "
            "dataset and Fourier feature configuration."
        )
        st.stop()


forecast = pd.DataFrame(
    {
        "Predicted minimum temperature (°C)": forecast_values
    },
    index=future_dates
)

forecast.index.name = "date"


# ==================================================
# 8. Forecast date range and plot
# ==================================================

st.caption(
    f"Forecast period: "
    f"{future_dates.min():%d %B %Y} to "
    f"{future_dates.max():%d %B %Y} "
    f"({len(future_dates)} daily predictions)."
)

fig, ax = plt.subplots(figsize=(14, 6))

# Plot the last 365 historical observations.
recent_history = history["temperature"].tail(365)

ax.plot(
    recent_history.index,
    recent_history.values,
    label="Historical temperature",
    linewidth=1.2
)

ax.plot(
    forecast.index,
    forecast["Predicted minimum temperature (°C)"],
    label="Forecast",
    linewidth=1.5
)

ax.axvline(
    forecast_start,
    linestyle="--",
    linewidth=1,
    label="Forecast starts"
)

ax.set_title(
    "Daily Minimum Temperature: History and Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ==================================================
# 9. Monthly forecast summary
# ==================================================

st.subheader("Monthly Forecast Summary")

monthly_summary = (
    forecast["Predicted minimum temperature (°C)"]
    .resample("MS")
    .agg(["mean", "min", "max"])
)

monthly_summary.columns = [
    "Mean forecast (°C)",
    "Minimum forecast (°C)",
    "Maximum forecast (°C)"
]

st.dataframe(
    monthly_summary.round(2),
    use_container_width=True
)


# ==================================================
# 10. Daily forecast table
# ==================================================

st.subheader("Daily Forecast Values")

st.dataframe(
    forecast.round(2),
    use_container_width=True
)


# ==================================================
# 11. Download forecast as CSV
# ==================================================

csv_data = forecast.reset_index().to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Daily Forecast as CSV",
    data=csv_data,
    file_name=(
        f"melbourne_temperature_forecast_"
        f"{months_ahead}_months.csv"
    ),
    mime="text/csv"
)


# ==================================================
# 12. Model information
# ==================================================

with st.expander("Model Details"):

    st.write("**Model:** Fourier + ARIMA(2, 0, 1)")

    st.write(
        "**Model loading:** Uses the saved fitted model "
        "from fourier_arima_model.pkl; it does not fit "
        "the model again when generating predictions."
    )

    st.write(
        "**Training data:** The model file was created using "
        "the historical observations through 31 December 1990."
    )

    st.write(
        "**Seasonality:** Two annual Fourier harmonics "
        "with a period of 365.25 days."
    )

    st.write(
        "**Forecasting:** Multi-step prediction without "
        "using actual future temperatures."
    )

st.info(
    "These are multi-step future predictions. They differ from "
    "the rolling one-day-ahead forecasts used to evaluate models "
    "on the 1990 test period."
)
