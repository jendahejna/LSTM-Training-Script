import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold  # Přidán KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.mixed_precision import set_global_policy
from keras_tuner import HyperModel, BayesianOptimization

# --- Nastavení Mixed Precision ---
# Povolí globální politiku mixed_float16 pro zrychlení tréninku a úsporu paměti GPU.
# Výstupní vrstvy zůstávají v float32 pro přesnost.
set_global_policy('mixed_float16')

# --- Konfigurace GPU ---
# Detekce a konfigurace GPU pro TensorFlow. Povolí dynamický růst paměti.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Povolení dynamického růstu paměti GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("Žádná GPU nebyla nalezena.")

# Konstanty
# Scaler nebude globální, bude se inicializovat lokálně tam, kde je potřeba pro zamezení data leakage.
base_data_dir = r'Training_sets'
model_path = './models'
output_path = './output'
scaler_save_path = '.'  # Scaler se uloží do stejného adresáře, kde je skript

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
    # plt.show() # Komentuji plt.show() pro automatizované spouštění, můžete odkomentovat pro vizualizaci


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


# Funkce pro standardizaci dat (používá se lokálně pro každý fold nebo pro tuner)
def _scale_data_and_save_scaler(X_train, X_test, scaler_path=None):
    """
    Standardizuje trénovací a testovací data a volitelně uloží scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    return X_train_scaled, X_test_scaled, scaler


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
# Tato třída definuje strukturu modelu a hyperparametry, které budou laděny Bayesovskou optimalizací.
class LSTMTuningHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        # Vstupní vrstva pro LSTM, kde input_dim je počet featur.
        inputs = Input(shape=(self.input_dim, 1))

        # Optimalizujte počet jednotek v první LSTM vrstvě.
        # Rozsah 64 až 256 s krokem 32.
        lstm1_units = hp.Int('lstm1_units', min_value=64, max_value=256, step=32)
        # První LSTM vrstva
        x = LSTM(lstm1_units, return_sequences=True)(inputs)

        # Optimalizujte počet jednotek v druhé LSTM vrstvě.
        # Rozsah 64 až 256 s krokem 32.
        lstm2_units = hp.Int('lstm2_units', min_value=64, max_value=256, step=32)
        # Druhá LSTM vrstva
        x = LSTM(lstm2_units)(x)

        # Výstupní Dense vrstva s jedním výstupem.
        # dtype='float32' je klíčové pro přesnost výstupů při mixed precision.
        outputs = Dense(1, dtype='float32')(x)

        # Optimalizujte learning rate pro Adam optimalizátor.
        # Tuner vybere jednu z předdefinovaných hodnot.
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean Squared Error jako ztrátová funkce
            metrics=['mae']  # Mean Absolute Error jako metrika
        )
        return model


# Funkce pro sestavení modelu s danými hyperparametry (pro použití v křížové validaci)
def build_model_with_hps(input_dim, lstm1_units, lstm2_units, learning_rate):
    inputs = Input(shape=(input_dim, 1))
    x = LSTM(lstm1_units, return_sequences=True)(inputs)
    x = LSTM(lstm2_units)(x)
    outputs = Dense(1, dtype='float32')(x)

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
    # --- Zajištění existence výstupních adresářů na začátku ---
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

    # Načtení a předzpracování celého datasetu
    df_full = _load_and_preprocess(dataset_filenames)
    X_full, y_full = _split_data(df_full)

    # Rozdělení dat pro Bayesovskou optimalizaci a finální test
    # Toto je počáteční rozdělení před KFold, aby tuner měl svou testovací sadu.
    X_train_opt, X_test_final, y_train_opt, y_test_final = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, shuffle=True
    )

    # Škálování dat pro Bayesovskou optimalizaci a uložení scaleru
    X_train_opt_scaled, X_test_final_scaled, initial_scaler = _scale_data_and_save_scaler(
        X_train_opt, X_test_final, os.path.join(scaler_save_path, 'scaler.joblib')
    )

    print(f"Počet použitých features: {X_train_opt.shape[1]}")

    # Reshape dat pro LSTM pro tuner
    X_train_opt_reshaped = X_train_opt_scaled.reshape((X_train_opt_scaled.shape[0], X_train_opt_scaled.shape[1], 1))
    X_test_final_reshaped = X_test_final_scaled.reshape((X_test_final_scaled.shape[0], X_test_final_scaled.shape[1], 1))

    # Inicializace HyperModelu s rozměry vstupních dat
    hypermodel = LSTMTuningHyperModel(input_dim=X_train_opt_reshaped.shape[1])

    # Konfigurace Bayesovské optimalizace
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_mae',
        max_trials=20,
        executions_per_trial=1,
        directory='my_tuning_results',
        project_name='lstm_temperature_prediction'
    )

    print("Spouštím Bayesovskou optimalizaci hyperparametrů...")
    tuner.search(X_train_opt_reshaped, y_train_opt, epochs=100, batch_size=128,
                 validation_split=0.2,  # Tuner si vezme 20% dat z X_train_opt pro validaci
                 callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)],
                 verbose=1)

    # Získání nejlepších hyperparametrů a nejlepšího modelu nalezeného tunerem
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    # Upozornění: best_model zde je jen jeden model, který byl "nejlepší" během tuningu.
    # Není to model, který prošel křížovou validací.
    best_model_from_tuner = tuner.get_best_models(num_models=1)[0]

    print(f"\nNejlepší hyperparametry nalezené Bayesovskou optimalizací: {best_hp.values}")

    # --- K-Fold Cross-Validation s nejlepšími hyperparametry ---
    print("\nSpouštím K-Fold křížovou validaci s nejlepšími nalezenými hyperparametry...")

    n_splits = 5  # Počet foldů pro křížovou validaci
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []
    # Iterujeme přes foldy na původních, nenaškálovaných datech X_full
    for fold_idx, (train_index, test_index) in enumerate(kf.split(X_full)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        X_train_fold_original, X_test_fold_original = X_full.iloc[train_index], X_full.iloc[test_index]
        y_train_fold, y_test_fold = y_full.iloc[train_index], y_full.iloc[test_index]

        # Škálování dat uvnitř každého foldu, aby se zabránilo data leakage
        scaler_fold = StandardScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_original)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold_original)

        # Reshape dat pro LSTM
        X_train_fold_reshaped = X_train_fold_scaled.reshape(
            (X_train_fold_scaled.shape[0], X_train_fold_scaled.shape[1], 1))
        X_test_fold_reshaped = X_test_fold_scaled.reshape((X_test_fold_scaled.shape[0], X_test_fold_scaled.shape[1], 1))

        # Vytvoření nové instance modelu pro každý fold s nejlepšími hyperparametry
        fold_model = build_model_with_hps(
            input_dim=X_train_fold_reshaped.shape[1],
            lstm1_units=best_hp.get('lstm1_units'),
            lstm2_units=best_hp.get('lstm2_units'),
            learning_rate=best_hp.get('learning_rate')
        )

        # Připravit tf.data datasety pro trénink a testování ve foldu
        train_ds_fold = tf.data.Dataset.from_tensor_slices((X_train_fold_reshaped, y_train_fold)).shuffle(
            buffer_size=1024).batch(128).prefetch(tf.data.AUTOTUNE)
        test_ds_fold = tf.data.Dataset.from_tensor_slices((X_test_fold_reshaped, y_test_fold)).batch(128).prefetch(
            tf.data.AUTOTUNE)

        # Trénink modelu ve foldu
        history_fold = fold_model.fit(
            train_ds_fold,
            epochs=100,  # Maximální počet epoch, EarlyStopping bude řídit zastavení
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                       ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)],
            validation_data=test_ds_fold,  # Testovací fold slouží jako validační pro early stopping
            verbose=0  # Snížit verbosity pro čistější výstup
        )

        # Vyhodnocení modelu ve foldu
        y_pred_fold = fold_model.predict(test_ds_fold).flatten()
        metrics_fold = _compute_metrics(y_test_fold, y_pred_fold)
        fold_metrics.append(metrics_fold)
        print(
            f"Metriky Fold {fold_idx + 1}: MAE={metrics_fold[0]:.4f}, MSE={metrics_fold[1]:.4f}, RMSE={metrics_fold[2]:.4f}, R2={metrics_fold[3]:.4f}")

    # Průměrné metriky napříč všemi foldy
    avg_metrics = np.mean(fold_metrics, axis=0)
    print(f"\nPrůměrné metriky K-Fold křížové validace ({n_splits} foldů):")
    print(f"MAE: {avg_metrics[0]:.4f}")
    print(f"MSE: {avg_metrics[1]:.4f}")
    print(f"RMSE: {avg_metrics[2]:.4f}")
    print(f"R2: {avg_metrics[3]:.4f}")

    # Zápis průměrných metrik křížové validace do souboru
    with open(os.path.join(output_path, 'kfold_cross_validation_metrics.txt'), 'w') as f:
        f.write(
            f"Průměrné metriky K-Fold křížové validace ({n_splits} foldů) (MAE, MSE, RMSE, R2): {avg_metrics.tolist()}\n")
        f.write(f"Použité hyperparametry: {best_hp.values}\n")
    print(f"Průměrné metriky K-Fold uloženy do: {os.path.join(output_path, 'kfold_cross_validation_metrics.txt')}")

    # --- Vyhodnocení nejlepšího modelu (z Bayesovské optimalizace) na původní finální testovací sadě ---
    # Toto je oddělené od K-Fold a představuje vyhodnocení modelu na datech, která tuner nikdy neviděl.
    print("\n--- Vyhodnocení nejlepšího modelu (z Bayesovské optimalizace) na původní finální testovací sadě ---")

    # Pro predikci na final test set použijeme scaler, který byl fitován na X_train_opt
    X_test_final_scaled_for_pred = initial_scaler.transform(X_test_final)
    X_test_final_reshaped_for_pred = X_test_final_scaled_for_pred.reshape(
        (X_test_final_scaled_for_pred.shape[0], X_test_final_scaled_for_pred.shape[1], 1))

    test_ds_final = tf.data.Dataset.from_tensor_slices((X_test_final_reshaped_for_pred, y_test_final)).batch(
        128).prefetch(tf.data.AUTOTUNE)

    y_pred_final_test = best_model_from_tuner.predict(test_ds_final).flatten()
    final_test_metrics = _compute_metrics(y_test_final, y_pred_final_test)
    print('Výsledné metriky nejlepšího LSTM modelu na původní finální testovací sadě (MAE, MSE, RMSE, R2):',
          final_test_metrics)

    # Uložení nejlepšího modelu z Bayesovské optimalizace
    best_model_from_tuner.save(os.path.join(model_path, 'best_tuned_lstm_model.keras'))
    print(
        f"Nejlepší vyladěný model (z Bayesovské optimalizace) uložen do: {os.path.join(model_path, 'best_tuned_lstm_model.keras')}")

    # Vykreslení výsledků nejlepšího modelu na původní testovací sadě
    plot_the_results(y_test_final, y_pred_final_test, title='LSTM Model (tuned_final_test)', save=True)
    print(
        f"Výsledný graf pro finální testovací sadu uložen do: {os.path.join(output_path, 'figs', 'LSTM Model (tuned_final_test).png')}")

    # Zápis metrik a nejlepších hyperparametrů do souboru (pro původní testovací sadu)
    with open(os.path.join(output_path, 'tuned_lstm_model_metrics.txt'), 'w') as f:
        f.write(
            f"Nejlepší vyladěné LSTM metriky na původní finální testovací sadě (MAE, MSE, RMSE, R2): {final_test_metrics}\n")
        f.write(f"Nejlepší hyperparametry: {best_hp.values}\n")
    print(f"Metriky finální testovací sady uloženy do: {os.path.join(output_path, 'tuned_lstm_model_metrics.txt')}")