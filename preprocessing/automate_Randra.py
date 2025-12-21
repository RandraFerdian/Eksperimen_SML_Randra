import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- KONFIGURASI PATH (VERSI ANTI ERROR) ---
# Iki trik ben python ngerti posisi file iki neng ngendi
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Folder 'preprocessing'
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)              # Folder 'Eksperimen_SML_Randra'

# Path sing bener (gabungke root karo folder data)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "btc_data_raw", "btc_usd_5y.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "btc_data_preprocessed")

def load_data(path):
    """
    Fungsi kanggo mbukak data mentah.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data ora ketemu neng: {path}. Pastike script 'setup' wis dijalanke.")
    
    print(f"📂 Loading data soko: {path}...")
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def feature_engineering_financial(df):
    """
    Fungsi 'Daging': Nggawe indikator teknikal (EMA, MACD, RSI, BB, OBV).
    Persis karo sing ono neng Notebook Eksperimen revisi terakhir.
    """
    print("🛠️ Lagi ngeracik indikator trading...")
    
    # 1. EMA (Short Term - Aggressive)
    # Nggo sinyal Entry/Exit cepet
    df['EMA_7'] = df['Close'].ewm(span=7, adjust=False).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

    # 2. SMA (Long Term - Trend)
    # Nggo filter tren besar
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()

    # 3. RSI (Momentum)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 4. MACD (Trend Momentum)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. Bollinger Bands (Volatility)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (2 * std_20)
    df['BB_Lower'] = sma_20 - (2 * std_20)

    # 6. OBV (Volume Validation)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # --- MEMBUAT TARGET ---
    # 1 = Buy (Harga besok naik), 0 = Sell (Harga besok turun/tetap)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Bersih-bersih NaN
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"🧹 Cleaning data: Dibuang {before - after} baris (efek rolling window).")
    
    return df

def split_and_save(df):
    """
    Fungsi kanggo mbagi data lan nyimpen dadi CSV siap latih.
    """
    # Seleksi Fitur (Kudu podo karo Notebook!)
    features = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'EMA_7', 'EMA_14', 'SMA_50', 'SMA_200', 
        'RSI', 'MACD', 'MACD_Signal', 
        'BB_Upper', 'BB_Lower', 'OBV'
    ]
    
    X = df[features]
    y = df['Target']

    # Split Data (Tanpa Shuffle mergo Time Series)
    print("✂️ Mbagi data: 80% Training, 20% Testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Gabungke meneh kanggo disimpen
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)

    # Simpen
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    train_set.to_csv(f"{OUTPUT_PATH}/train.csv", index=False)
    test_set.to_csv(f"{OUTPUT_PATH}/test.csv", index=False)
    
    print(f"✅ SUKSES! File train.csv & test.csv wis kesimpen neng folder '{OUTPUT_PATH}'")

if __name__ == "__main__":
    try:
        # Pipeline Eksekusi
        data = load_data(RAW_DATA_PATH)
        data_clean = feature_engineering_financial(data)
        split_and_save(data_clean)
        print("\n🚀 Script Automate mlaku lancar jaya!")
    except Exception as e:
        print(f"\n❌ WADUH ERROR: {e}")