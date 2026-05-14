import pandas as pd
import numpy as np
import EnvTest
import LoadData_FordEngine
import Visualization
import Correlation
import Windowing
from Split import train_val_split
from sklearn.preprocessing import StandardScaler, RobustScaler


def main():
    # Environment Test
    EnvTest.env_test()

    # Load Data
    train_data, test_data = LoadData_FordEngine.load_data()
    train_data, valid_data = train_val_split(train_data, val_ratio=0.2, 
                                             stratify=False, shuffle=False, random_state=0)

    x_train, y_train = train_data
    x_valid, y_valid = valid_data
    x_test, y_test = test_data

    y_train[y_train==-1] = 0
    y_valid[y_valid==-1] = 0
    y_test[y_test==-1] = 0

    train_data = [x_train, y_train]
    valid_data = [x_valid, y_valid]
    test_data = [x_test, y_test]

    # Class Balance Check
    Visualization.plot_class_distribution(
        train_data, valid_data, test_data,
        class_names={1: "normal", 0: "abnormal"},
        #save_path="class_distribution.png",
        show=False
    )

    # Data Correlation
    dataCorr = Correlation.correlation(
        x_train,
        feature_names=[f"sensor_{i+1}" for i in range(x_train.shape[1])],
    )
    Visualization.plot_correlation_matrix(
        dataCorr,
        #save_path="sensor_correlation_spearman.png",
        show=False)
    
    # Data Normalization
    scaler = RobustScaler()
    scaler.fit(x_train)
    x_train_norm = scaler.transform(x_train)
    x_valid_norm = scaler.transform(x_valid)
    x_test_norm = scaler.transform(x_test)

    Visualization.plot_normalization_comparison(
        x_train, x_train_norm,
        n_samples=5, sample_indices=[0, 700, 1400, 2100, 2800],
        n_sensors=5, sensor_indices=[0, 100, 200, 300, 400],
        #save_path="normalization_robust.png",
        show=False,
    )

    # Model Selection: "LR" | "XGB" | "RNN" | "CNN"
    MODEL = "CNN"
    LR_TOP_K = 50              # LR feature selection 크기. None이면 selection 안 함
    CORR_METHOD = 'pearson'    # 'pearson' | 'spearman' | 'kendall'  (LR feature selection 기준)
    LR_PENALTY = 'l1'          # 'l1' | 'l2'
    LR_C = 1.0                 # regularization 강도 (작을수록 강함)

    # RNN/CNN windowing config (LR/XGB는 무시됨)
    USE_WINDOWING = False       # True: 시간축 살리는 rolling window | False: 기존 expand_dims
    WINDOW_SIZE = 1
    WINDOW_STRIDE = 1
    LABEL_POS = 'last'         # 'last' | 'center' | 'majority'

    if   MODEL == "LR":  from Models import LogisticRegression as model
    elif MODEL == "XGB": from Models import XGBoost as model
    elif MODEL == "RNN": from Models import RNN as model
    elif MODEL == "CNN": from Models import CNN as model
    else: raise ValueError(f"Unknown model: {MODEL}")

    # 모델 입력 prep
    #  - RNN/CNN + USE_WINDOWING: (T, S) -> (N, W, S), 라벨도 window 단위로 변환
    #  - RNN/CNN + windowing off: 기존 expand_dims (sensor 축을 timesteps로 잘못 해석하는 baseline)
    #  - LR/XGB: 2D 그대로
    if MODEL in ("RNN", "CNN") and USE_WINDOWING:
        x_tr, y_tr = Windowing.make_windows(x_train_norm, y_train, WINDOW_SIZE, WINDOW_STRIDE, LABEL_POS)
        x_va, y_va = Windowing.make_windows(x_valid_norm, y_valid, WINDOW_SIZE, WINDOW_STRIDE, LABEL_POS)
        x_te, y_te = Windowing.make_windows(x_test_norm,  y_test,  WINDOW_SIZE, WINDOW_STRIDE, LABEL_POS)
        print(f"[Windowing] enabled: W={WINDOW_SIZE}, stride={WINDOW_STRIDE}, label={LABEL_POS}")
    elif MODEL in ("RNN", "CNN"):
        x_tr = np.expand_dims(x_train_norm, -1)
        x_va = np.expand_dims(x_valid_norm, -1)
        x_te = np.expand_dims(x_test_norm, -1)
        y_tr, y_va, y_te = y_train, y_valid, y_test
        print("[Windowing] disabled — using expand_dims path")
    else:
        x_tr, x_va, x_te = x_train_norm, x_valid_norm, x_test_norm
        y_tr, y_va, y_te = y_train, y_valid, y_test

    # Feature ↔ Target Correlation (전체 500, 정렬된 bar chart)
    target_corrs = Correlation.correlation(x_train_norm, y_train, method=CORR_METHOD)
    Visualization.plot_topk_correlation(
        target_corrs,
        top_k=LR_TOP_K if MODEL == "LR" else None,
        #save_path="sensor_label_correlation.png",
        show=False,
    )

    # Top-K Selected Feature Correlation (LR with selection)
    if MODEL == "LR" and LR_TOP_K is not None and LR_TOP_K < x_train_norm.shape[1]:
        top_idx = Correlation.top_k_indices(x_train_norm, y_train, LR_TOP_K)

    print("Input shape:", x_tr.shape, y_tr.shape, x_va.shape, y_va.shape)

    suffix = MODEL.lower()
    train_kwargs = {}
    if MODEL == "LR":
        train_kwargs = {
            "top_k": LR_TOP_K,
            "selection_method": CORR_METHOD,
            "penalty": LR_PENALTY,
            "C": LR_C,
        }
    elif MODEL in ("RNN", "CNN"):
        train_kwargs = {"save_path": f"model_best_{suffix}.keras"}
    trained, history = model.train(x_tr, y_tr, x_va, y_va, **train_kwargs)

    if MODEL == "LR":
        coef = trained.model.coef_
        n_nonzero = int(np.sum(coef != 0))
        n_tiny = int(np.sum(np.abs(coef) < 1e-4))
        n_total = int(coef.size)
        print(f"  [LR] coefficients: "
              f"non-zero {n_nonzero}/{n_total} ({100*n_nonzero/n_total:.1f}%), "
              f"|w|<1e-4 {n_tiny}/{n_total} ({100*n_tiny/n_total:.1f}%), "
              f"max|w|={np.abs(coef).max():.4f}")

    y_pred = model.predict(trained, x_te)
    y_score = model.predict_proba(trained, x_te)

    # Evaluation
    Visualization.plot_confusion_matrix(
        y_te, y_pred,
        class_names={1: "normal", 0: "abnormal"},
        save_path=f"confusion_matrix_{suffix}.png",
        show=True,
    )
    Visualization.plot_roc_curve(
        y_te, y_score,
        save_path=f"roc_curve_{suffix}.png",
        show=True)
    # save_path는 빼고 넘김 — learning curve의 5회 재학습이 메인 best checkpoint를 덮어쓰지 않게
    lc_train_kwargs = {k: v for k, v in train_kwargs.items() if k != "save_path"}
    Visualization.plot_learning_curve(
        model.train, model.predict,
        x_tr, y_tr, x_va, y_va,
        train_kwargs=lc_train_kwargs,
        save_path=f"learning_curve_{suffix}.png",
        show=True,
    )
    Visualization.plot_training_history(
        history,
        save_path=f"training_loss_{suffix}.png",
        show=True)


if __name__ == "__main__":
    main()
