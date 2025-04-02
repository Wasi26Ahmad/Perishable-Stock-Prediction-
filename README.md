# Perishable Stock Prediction Pipeline Installation
## To set up the environment and install dependencies, run:
```sh
pip install -r requirements.txt
```
## Description of User-Defined Functions
```python
load_and_prepare_data(file_path)
```
- Loads and preprocesses the data
- Reads CSV file
- Sorts and pivots data
- Uses KNN imputation for missing values(You can use other methods if you wish)
- Normalizes data with MinMaxScaler
- Saves the scaler for future use
```python
create_sequences(data, seq_length=30)
```
- Converts the time-series data into sequences for LSTM training.
```python
build_model(hp)
```
- Defines an LSTM model with tunable hyperparameters.
```python
tune_and_train_model(X_train, y_train, X_test, y_test)
```
- Tunes the model using Keras Tuner (RandomSearch) and trains it with the best hyperparameters.
```python
update_model(new_data_path, model_path, scaler_path)
```
- Loads new data, updates the trained model, and saves it.
## How to Use Each Function in main.py
### Load Data and Preprocess:
```python
df_scaled = load_and_prepare_data("dataset.csv") 
```
### Create Sequences:
```python
X, y = create_sequences(df_scaled, seq_length=30) 
```
### Train and Tune Model:
```python
best_model = tune_and_train_model(X_train, y_train, X_test, y_test)
```
### Update Model with New Data:
```python
update_model("new_data.csv")
```
### Using Different Tuning Methods (e.g., PSO)
- To replace Keras Tuner with Particle Swarm Optimization (PSO), update the tune_and_train_model function
## Running the Code
To execute the pipeline, run:
```sh
python main.py
```
