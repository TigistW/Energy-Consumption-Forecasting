# 🔮 Future Forecasting with Autoregressive LSTM/GRU

This Streamlit web app allows you to upload a preprocessed time series CSV dataset with 11 specific features and generate future forecasts using two deep learning models: LSTM and GRU. You can select which feature to predict and compare the forecasts visually and in tabular form.

---

## Features

- Upload a CSV file containing the required features.
- Select the target feature to predict from the 11 available.
- Configure the input sequence length (window size) and number of future steps to predict.
- Autoregressive forecasting using pretrained LSTM and GRU PyTorch models.
- Visualization of historical data alongside forecasts from both models.
- Side-by-side comparison of forecasted values.

---

## Required Features in CSV

Your uploaded CSV must contain these columns:

- `Global_active_power`
- `Global_reactive_power`
- `Voltage`
- `Global_intensity`
- `Sub_metering_1`
- `Sub_metering_2`
- `Sub_metering_3`
- `hour`
- `dayofweek`
- `month`
- `is_weekend`

---

## Getting Started

### Prerequisites

- Python 3.8 or above
- PyTorch
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib

Install dependencies via:

```bash
pip install torch streamlit pandas numpy scikit-learn matplotlib
````

### Models

Place your pretrained model weights (`lstm_model.pt` and `gru_model.pt`) in the `../models/` directory relative to the app script.

---

## Running the App

Run the Streamlit app with:

```bash
streamlit run your_app_script.py
```

---

## Usage

1. Upload a CSV file with the required features.
2. Select the data slice to use (start and end row indices).
3. Choose the feature you want to predict.
4. Set the input sequence length and number of future steps to predict.
5. View the forecast plot comparing LSTM and GRU predictions.
6. Review forecasted values in the comparison table.

---

## Notes

* The app uses MinMaxScaler internally to scale features before prediction.
* The forecasting is autoregressive: each predicted step is fed as input to predict the next.
* Make sure your CSV contains no missing values for the required features.

---

## License

This project is open source under the MIT License.
