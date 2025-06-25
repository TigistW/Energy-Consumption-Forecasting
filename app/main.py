import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ----------------------
# LSTM Model Definition
# ----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1)

# ----------------------
# GRU Model Definition
# ----------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(-1)

# ----------------------
# Streamlit Interface
# ----------------------
st.set_page_config(page_title="🔮 Future Forecasting", layout="wide")
st.title("🔮 Future Forecast with Autoregressive LSTM/GRU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uploaded_file = st.file_uploader("Upload preprocessed time series CSV", type=["csv"])

required_features = [
    'Global_active_power', 'Global_reactive_power', 'Voltage',
    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
    'hour', 'dayofweek', 'month', 'is_weekend'
]

if uploaded_file is not None:
    uploaded_file.seek(0)
    df_full = pd.read_csv(uploaded_file, usecols=required_features)

    st.write(f"Total rows in uploaded data: {len(df_full)}")

    start_row = st.number_input(
        "Select start row index", min_value=0, max_value=len(df_full)-1, value=max(0, len(df_full)-100)
    )
    end_row = st.number_input(
        "Select end row index", min_value=start_row+1, max_value=len(df_full), value=len(df_full)
    )

    df = df_full.iloc[start_row:end_row].reset_index(drop=True)

    st.write("✅ Preview of Selected Data")
    st.dataframe(df.head())

    if not all(feat in df.columns for feat in required_features):
        st.error(f"Uploaded CSV must contain these columns:\n{', '.join(required_features)}")
    else:
        # Add dropdown for target feature selection
        target_feature = st.selectbox("Select Target Feature to Predict", required_features, index=0)
        target_idx = required_features.index(target_feature)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[required_features])
        scaled_df = pd.DataFrame(scaled, columns=required_features)

        window_size = st.slider("Input Sequence Length (Window Size)", 12, 72, 24)
        future_steps = st.slider("Number of Future Steps to Predict", 1, 72, 24)

        if len(scaled_df) < window_size:
            st.warning("Not enough rows for selected window size.")
        else:
            input_seq = torch.tensor(scaled_df[-window_size:].values, dtype=torch.float32).unsqueeze(0).to(device)

            # Load LSTM model
            lstm_model = LSTMModel(input_dim=len(required_features))
            lstm_model.load_state_dict(torch.load("/models/lstm_model.pt", map_location=device))
            lstm_model.to(device).eval()

            # Load GRU model
            gru_model = GRUModel(input_dim=len(required_features))
            gru_model.load_state_dict(torch.load("/models/gru_model.pt", map_location=device))
            gru_model.to(device).eval()

            def autoregressive_forecast(model, input_seq, steps, target_idx):
                preds = []
                seq = input_seq.clone()
                with torch.no_grad():
                    for _ in range(steps):
                        pred = model(seq)
                        preds.append(pred.item())
                        next_row = seq[:, -1, :].clone()
                        next_row[:, target_idx] = pred  # Replace selected target feature with predicted value
                        seq = torch.cat((seq[:, 1:, :], next_row.unsqueeze(1)), dim=1)
                return preds

            # Forecast both models
            lstm_preds_scaled = autoregressive_forecast(lstm_model, input_seq, future_steps, target_idx)
            gru_preds_scaled = autoregressive_forecast(gru_model, input_seq, future_steps, target_idx)

            # Inverse transform predictions (only target feature)
            dummy_lstm = np.zeros((future_steps, len(required_features)))
            dummy_lstm[:, target_idx] = lstm_preds_scaled
            lstm_preds = scaler.inverse_transform(dummy_lstm)[:, target_idx]

            dummy_gru = np.zeros((future_steps, len(required_features)))
            dummy_gru[:, target_idx] = gru_preds_scaled
            gru_preds = scaler.inverse_transform(dummy_gru)[:, target_idx]

            # Plot both forecasts + history
            st.subheader(f"📈 Forecast Plot: LSTM vs GRU for '{target_feature}'")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(range(len(df)), df[target_feature], label="History", color="black")
            ax.plot(range(len(df), len(df) + future_steps), lstm_preds, label="LSTM Forecast", color="blue")
            ax.plot(range(len(df), len(df) + future_steps), gru_preds, label="GRU Forecast", color="red")
            ax.legend()
            st.pyplot(fig)

            # Show forecast tables side by side
            st.subheader(f"📋 Forecasted Values Comparison for '{target_feature}'")
            df_forecasts = pd.DataFrame({
                "LSTM Prediction": lstm_preds,
                "GRU Prediction": gru_preds,
            })
            st.write(df_forecasts)

else:
    st.info("Upload a CSV file containing the required 11 feature columns.")
