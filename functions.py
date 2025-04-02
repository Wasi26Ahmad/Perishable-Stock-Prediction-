import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import joblib
from keras_tuner import RandomSearch

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.sort_values(by=['date', 'product'], inplace=True)
    df_pivot = df.pivot(index='date', columns='product', values='demand')
    
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df_pivot)
    df_imputed = pd.DataFrame(df_imputed, columns=df_pivot.columns, index=df_pivot.index)
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_imputed)
    joblib.dump(scaler, "scaler.pkl")
    
    return df_scaled

def create_sequences(data, seq_length=30):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)

def build_model(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('units1', min_value=32, max_value=128, step=32), return_sequences=True, input_shape=(30, None)))
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('units2', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('units3', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('units4', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout4', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('units5', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dropout(hp.Float('dropout5', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(None))
    model.compile(optimizer='adam', loss='mse')
    return model

def tune_and_train_model(X_train, y_train, X_test, y_test):
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=5,
        executions_per_trial=1,
        directory='hyperparam_tuning',
        project_name='perishable_stock'
    )
    
    tuner.search(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    best_model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    best_model.save("perishable_stock_model.h5")
    return best_model

def update_model(new_data_path, model_path="perishable_stock_model.h5", scaler_path="scaler.pkl"):
    df_new_scaled = load_and_prepare_data(new_data_path)
    X_new, y_new = create_sequences(df_new_scaled, seq_length=30)
    
    model = keras.models.load_model(model_path)
    model.fit(X_new, y_new, epochs=5, batch_size=16)
    model.save(model_path)
    print("Model updated with new data.")
