import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# --- KONFIGURASI PATH (Anti Error) ---
# Nggoleki folder di mana script iki berada (ROOT)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "btc_data_raw")
# Jeneng file tetep 'btc_usd_5y.csv' wae ben ora perlu ngubah script liyane
# Senajan isine saiki wis luwih seko 5 taun.
FILE_PATH = os.path.join(DATA_FOLDER, "btc_usd_5y.csv") 

def update_bitcoin_data():
    print(f"\n🔄 MEMULAI DOWNLOAD DATA BITCOIN (MAX HISTORY)...")
    print(f"🕒 Waktu Server: {datetime.now()}")
    
    # 1. Download Data SAK-ANA-NE (Max)
    # period="max" artine njupuk kabeh data sing diduweni Yahoo Finance
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="max", interval="1d")
    
    if df.empty:
        print("❌ GAGAL! Ora iso download data. Cek internetmu.")
        return

    # 2. Rapikan Data
    # Mung njupuk kolom penting
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    
    # Format tanggal (ngilangke jam/menit)
    df['Date'] = df['Date'].dt.date
    
    # 3. Simpen / Timpa File Lawas
    # Gawe folder nek durung ono
    os.makedirs(DATA_FOLDER, exist_ok=True)
    
    # Simpen CSV
    df.to_csv(FILE_PATH, index=False)
    
    # --- Laporan Statistik ---
    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]
    total_rows = len(df)
    
    print(f"\n✅ SUKSES! Data wis kesimpen.")
    print(f"📂 Lokasi File: {FILE_PATH}")
    print(f"📊 Total Data: {total_rows} baris.")
    print(f"🗓️ Periode Data: {start_date} s/d {end_date}")
    print("🚀 Siap kanggo Preprocessing & Training!")

if __name__ == "__main__":
    update_bitcoin_data()