import numpy as np
import pandas as pd


def correlation(x, y=None, feature_names=None, method='pearson'):
    """
    y가 None이면 (n_features × n_features) DataFrame, 주어지면 (n_features,) numpy array.

    - DataFrame은 plot_correlation_matrix 등 시각화에 사용
    - numpy array는 feature selection 등 수치 처리에 사용
    """
    if y is None:
        names = feature_names or [f"col_{i}" for i in range(x.shape[1])]
        return pd.DataFrame(x, columns=names).corr(method=method, min_periods=1)

    return np.nan_to_num(
        pd.DataFrame(x).corrwith(pd.Series(y), method=method).values,
        nan=0.0,
    )


def top_k_indices(x, y, k, method='pearson'):
    """y와의 |correlation|이 큰 상위 k개 feature 인덱스를 반환."""
    corrs = np.abs(correlation(x, y, method=method))
    return np.argsort(corrs)[-k:]
