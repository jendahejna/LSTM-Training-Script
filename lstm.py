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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Žádná GPU nebyla nalezena.")

# Konstanty
scaler = StandardScaler()
model_path = './models'
output_path = './output'
PLOT_WIDTH = 252.0
FONT_SIZE = 8.0
FONT_FAMILY = "Times New Roman"
LINE_WIDTH = 0.8

# Funkce pro vykreslení výsledků
def plot_the_results(y_test, y_pred, title='', save=False, rasterized=False, format='png'):
    """
    Vykreslí porovnání skutečných a predikovaných hodnot.
    """
    if save and not os.path.exists('figs'):
        os.makedirs('figs')

    plt.figure()
    # Pokud y_test je pandas Series, převede se na numpy pole
    temp_npy = y_test.to_numpy() if isinstance(y_test, pd.Series) else y_test
    plt.scatter(temp_npy, y_pred, s=0.2, color='k', rasterized=rasterized)
    plt.xlabel('Skutečná teplota [°C]', fontsize=FONT_SIZE)
    plt.ylabel('Predikovaná teplota [°C]', fontsize=FONT_SIZE)
    plt.plot([min(temp_npy), max(temp_npy)], [min(temp_npy), max(temp_npy)], color='red', linestyle='--')
    plt.tight_layout()

    if save:
        plt.savefig(f'figs/{title}.{format}', format=format, dpi=1200, bbox_inches='tight')
    plt.show()

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

        # Převod 'Timestamp' na datetime a vytvoření pomocných sloupců
        df_tmp['Timestamp'] = pd.to_datetime(df_tmp['Timestamp'])
        df_tmp["Day"] = df_tmp["Timestamp"].dt.dayofyear
        df_tmp["Hour"] = df_tmp["Timestamp"].dt.hour

        # Vybereme pouze sloupce, které budeme používat při tréninku
        df_tmp = df_tmp[['Temperature_MW', 'sun', 'Hour', 'Day', 'Signal', 'Azimuth', 'Temperature_Meteo', 'Latitude', 'Longitude', 'Technology', 'Elevation']]
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
    joblib.dump(scaler, 'scaler.joblib')
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

# Funkce pro vytvoření LSTM modelu
def build_model(input_shape):
    """
    Vytvoří LSTM model se dvěma LSTM vrstvami (160 a 200 jednotek) a jednou Dense vrstvou.
    Poslední Dense vrstva je explicitně nastavena na float32, aby se zajistila správná přesnost výstupu při mixed precision.
    """
    inputs = Input(shape=input_shape)
    x = LSTM(160, return_sequences=True)(inputs)
    x = LSTM(200)(x)
    outputs = Dense(1, dtype='float32')(x)  # Nutné pro kompatibilitu s mixed precision
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
        loss='mse',
        metrics=['mae']
    )
    return model

# Funkce pro trénink modelu s využitím tf.data pipeline
def train_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=128, save_model=True, plot_results=True):
    """
    Připraví data, vytvoří model, natrénuje jej pomocí tf.data pipeline a vyhodnotí výkon.
    """
    # Reshape dat pro LSTM: (vzorky, kroky, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Další rozdělení trénovacích dat na trénovací a validační sadu
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )

    # Vytvoření tf.data datasetů pro trénink, validaci a test
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Vytvoření modelu
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)

    # Callbacky pro optimalizaci tréninku
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    # Trénink modelu
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Predikce na testovací datech
    y_pred = model.predict(test_ds).flatten()

    # Uložení modelu, pokud je požadováno
    if save_model:
        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)
        model.save(os.path.join(model_path, 'best_lstm_model.keras'))

    # Vykreslení výsledků, pokud je požadováno
    if plot_results:
        plot_the_results(y_test, y_pred, title='LSTM Model (160/200)', save=save_model)

    metrics = _compute_metrics(y_test, y_pred)
    return model, history, metrics, y_pred

# Hlavní část programu
if __name__ == '__main__':
    # Dynamické načtení všech CSV souborů z adresáře 'data'
    dataset_dir = 'data'
    dataset_filenames = [
        os.path.join(dataset_dir, file)
        for file in os.listdir(dataset_dir)
        if file.endswith('.csv')
    ]

    df = _load_and_preprocess(dataset_filenames)
    X, y = _split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    X_train_scaled, X_test_scaled = _scale_data(X_train, X_test)

    print(f"Počet použitých features: {X_train.shape[1]}")
    print("Spouštím trénink LSTM modelu s pevně danými parametry (160 a 200 jednotek)...")

    model, history, lstm_metrics, y_pred = train_model(
        X_train_scaled, y_train, X_test_scaled, y_test,
        epochs=100,
        batch_size=128,  # Zvýšená velikost batch
        save_model=True,
        plot_results=True
    )

    print('Výsledné metriky LSTM modelu:', lstm_metrics)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, 'lstm_model_comparisons.txt'), 'w') as f:
        f.write(f"LSTM (160/200): {lstm_metrics}\n")
