import numpy as np


def make_windows(x, y, window_size, stride=1, label_pos='last'):
    """
    (T, S) 시계열을 (N_windows, W, S) 윈도우 형태로 변환.

    Args:
        x: shape (T, S) — T timesteps, S features (sensors)
        y: shape (T,) — per-timestep 라벨
        window_size: window 길이 W
        stride: 연속 window 사이 간격
        label_pos: window 라벨 유도 방식
            - 'last':     window의 마지막 timestep의 라벨
            - 'center':   window 중심 timestep의 라벨
            - 'majority': window 안 라벨 중 다수결

    Returns:
        x_w: shape (N_windows, W, S)
        y_w: shape (N_windows,)
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    T = x.shape[0]
    if T < window_size:
        raise ValueError(f"window_size ({window_size}) > T ({T})")

    starts = np.arange(0, T - window_size + 1, stride)
    x_w = np.stack([x[s:s + window_size] for s in starts])

    if label_pos == 'last':
        y_w = y[starts + window_size - 1]
    elif label_pos == 'center':
        y_w = y[starts + window_size // 2]
    elif label_pos == 'majority':
        y_w = np.array([np.bincount(y[s:s + window_size].astype(int)).argmax() for s in starts])
    else:
        raise ValueError(f"Unknown label_pos: {label_pos!r}. Use 'last' | 'center' | 'majority'")

    return x_w, y_w
