import os
import glob
import numpy as np
import pandas as pd

from Split import train_val_split

DATA_DIR = './dataset/PlatingBath/'
ERROR_LOT_FILE = 'Error Lot list.csv'
SENSOR_COLS = ['pH', 'Temp', 'Current', 'Voltage']


def _load_error_lots(path):
    """Error Lot list.csv → {date_str: set(lot_nums)} 형태로 변환."""
    df = pd.read_csv(path)
    error_lots = {}
    for _, row in df.iterrows():
        if pd.notna(row['LoT']):
            error_lots.setdefault(row['Date'], set()).add(int(row['LoT']))
    return error_lots


def _date_from_filename(fn):
    """'kemp-abh-sensor-2021.09.06.csv' → '2021-09-06'"""
    base = os.path.basename(fn)
    return base.replace('kemp-abh-sensor-', '').replace('.csv', '').replace('.', '-')


def _load_one_day(csv_path, error_lots_for_date):
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    x = df[SENSOR_COLS].values.astype(np.float32)
    lots = df['Lot'].values
    # 정상=1, 불량=0
    y = np.where(np.isin(lots, list(error_lots_for_date)), 0, 1).astype(np.int64)
    return x, y


def _gather_all(data_dir):
    error_lots = _load_error_lots(os.path.join(data_dir, ERROR_LOT_FILE))
    csv_files = sorted(glob.glob(os.path.join(data_dir, 'kemp-abh-sensor-*.csv')))

    X_list, y_list = [], []
    for csv_path in csv_files:
        date = _date_from_filename(csv_path)
        x, y = _load_one_day(csv_path, error_lots.get(date, set()))
        X_list.append(x)
        y_list.append(y)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def load_data(test_ratio=0.2, random_state=0):
    """
    Returns:
        (x_train, y_train), (x_test, y_test)
        x: (N, 4) float32, y: (N,) int64 (0=abnormal, 1=normal)
    """
    X, y = _gather_all(DATA_DIR)
    return train_val_split((X, y), val_ratio=test_ratio,
                           stratify=False, shuffle=True, random_state=random_state)
