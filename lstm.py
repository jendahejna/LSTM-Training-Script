import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from keras_tuner import HyperModel, BayesianOptimization

# --- Nastavení Mixed Precision ---
set_global_policy('mixed_float16')

# --- Konfigurace GPU ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Žádná GPU nebyla nalezena.")

# Konstanty
scaler = StandardScaler()
base_data_dir = r'Training_sets'
model_path = './models'
output_path = './output'
scaler_save_path = '.'

PLOT_WIDTH = 252.0
FONT_SIZE = 8.0
FONT_FAMILY = "Times New Roman"
LINE_WIDTH = 0.8


# Funkce pro vykreslení výsledků
def plot_the_results(y_test, y_pred, title='', save=False, rasterized=False, format='png'):
    """
    Vykreslí porovnání skutečných a predikovaných hodnot.
    """
    if save:
        # Zde už by adresáře 'output' a 'output/figs' měly existovat díky kontrole v main bloku.
        # Tato kontrola je zde spíše pojistka, pokud by se funkce volala izolovaně.
        figs_dir = os.path.join(output_path, 'figs')
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)

    plt.figure()
    temp_npy = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
    plt.scatter(temp_npy, y_pred, s=0.2, color='k', rasterized=rasterized)
    plt.xlabel('Skutečná teplota [°C]', fontsize=FONT_SIZE)
    plt.ylabel('Predikovaná teplota [°C]', fontsize=FONT_SIZE)
    plt.plot([min(temp_npy), max(temp_npy)], [min(temp_npy), max(temp_npy)], color='red', linestyle='--')
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(figs_dir, f'{title}.{format}'), format=format, dpi=1200, bbox_inches='tight')
    # plt.show()


# Funkce pro načtení a předzpracování dat ze všech CSV souborů v adresáři 'data'
def _load_and_preprocess(dataset_filenames):
    """
    Načte a předzpracuje data z CSV souborů.
    Pouze požadované sloupce se zahrnou do tréninku – ostatní jsou ignorovány.
    """
    required_columns = ['Temperature_MW', 'sun', 'Timestamp', 'Signal', 'Azimuth', 'Temperature_Meteo']
    dfs = []

    for filename in dataset_filenames:
        df_tmp = pd.read_csv(filename, delimiter=',')
        if not set(required_columns).issubset(df_tmp.columns):
            raise ValueError(f"Soubor {filename} postrádá některý z požadovaných sloupců: {required_columns}")

        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        df_tmp["Day"] = df_tmp["Timestamp"].dt.dayofyear
        df_tmp["Hour"] = df_tmp["Timestamp"].dt.hour

        df_tmp = df_tmp[
            ['Temperature_MW', 'sun', 'Hour', 'Day', 'Signal', 'Azimuth', 'Temperature_Meteo', 'Latitude', 'Longitude',
             'Technology', 'Elevation']]
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    return df


# Rozdělení dat na vstupy a cílovou proměnnou
def _split_data(df):
    """
    Rozdělí data na vstupní hodnoty (X) a cílovou proměnnou (y).
    """
    X = df.drop(['Temperature_Meteo'], axis=1)
    y = df['Temperature_Meteo']
    return X, y


# Standardizace dat
def _scale_data(X_train, X_test):
    """
    Standardizuje trénovací a testovací data.
    """
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(scaler_save_path, 'scaler.joblib'))
    return X_train_scaled, X_test_scaled


# Výpočet metrik pro hodnocení modelu
def _compute_metrics(y_test, y_pred):
    """
    Vypočítá MAE, MSE, RMSE a R2.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2


# --- Třída pro HyperModel pro Keras Tuner ---
class LSTMTuningHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        inputs = Input(shape=(self.input_dim, 1))

        lstm1_units = hp.Int('lstm1_units', min_value=64, max_value=256, step=32)
        x = LSTM(lstm1_units, return_sequences=True)(inputs)

        lstm2_units = hp.Int('lstm2_units', min_value=64, max_value=256, step=32)
        x = LSTM(lstm2_units)(x)

        outputs = Dense(1, dtype='float32')(x)

        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        return model


# Hlavní část programu
if __name__ == '__main__':
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'figs'), exist_ok=True)  

    dataset_filenames = []
    for root, dirs, files in os.walk(base_data_dir):
        for file in files:
            if file.endswith('.csv'):
                dataset_filenames.append(os.path.join(root, file))

    if not dataset_filenames:
        raise FileNotFoundError(
            f"V adresáři '{base_data_dir}' ani jeho podadresářích nebyly nalezeny žádné CSV soubory.")

    df = _load_and_preprocess(dataset_filenames)
    X, y = _split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train_scaled, X_test_scaled = _scale_data(X_train, X_test)

    print(f"Počet použitých features: {X_train.shape[1]}")

    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    hypermodel = LSTMTuningHyperModel(input_dim=X_train_reshaped.shape[1])

    tuner = BayesianOptimization(
        hypermodel,
        objective='val_mae',
        max_trials=20,
        executions_per_trial=1,
        directory='my_tuning_results',
        project_name='lstm_temperature_prediction'
    )

    print("Spouštím Bayesovskou optimalizaci hyperparametrů...")
    tuner.search(X_train_reshaped, y_train, epochs=100, batch_size=128,
                 validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
                 verbose=1)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    print(f"\nNejlepší hyperparametry nalezené Bayesovskou optimalizací: {best_hp.values}")

    test_ds = tf.data.Dataset.from_tensor_slices((X_test_reshaped, y_test))
    test_ds = test_ds.batch(128).prefetch(tf.data.AUTOTUNE)

    y_pred = best_model.predict(test_ds).flatten()

    lstm_metrics = _compute_metrics(y_test, y_pred)
    print('Výsledné metriky nejlepšího LSTM modelu na testovacích datech (MAE, MSE, RMSE, R2):', lstm_metrics)

    best_model.save(os.path.join(model_path, 'best_tuned_lstm_model.keras'))
    print(f"Nejlepší vyladěný model uložen do: {os.path.join(model_path, 'best_tuned_lstm_model.keras')}")

    plot_the_results(y_test, y_pred, title='LSTM Model (tuned)', save=True)
    print(f"Výsledný graf uložen do: {os.path.join(output_path, 'figs', 'LSTM Model (tuned).png')}")

    with open(os.path.join(output_path, 'tuned_lstm_model_metrics.txt'), 'w') as f:
        f.write(f"Nejlepší vyladěné LSTM metriky (MAE, MSE, RMSE, R2): {lstm_metrics}\n")
        f.write(f"Nejlepší hyperparametry: {best_hp.values}\n")