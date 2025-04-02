from functions import load_and_prepare_data, create_sequences, tune_and_train_model, update_model
import joblib
import numpy as np

data_path = "dataset.csv"#Load you data here
df_scaled = load_and_prepare_data(data_path)

seq_length = 30
X, y = create_sequences(df_scaled, seq_length)

split = int(0.8 * len(X))
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

best_model = tune_and_train_model(X_train, y_train, X_test, y_test)


scaler = joblib.load("scaler.pkl")#Load your scaler here
joblib.dump(scaler, "scaler.pkl")

print("Model training complete and saved.")
