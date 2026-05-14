import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression as _LR
from sklearn.metrics import log_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import PolynomialFeatures
from Correlation import top_k_indices


class _LRPipeline:
    """LR + 선택적 feature selection + 선택적 polynomial expansion 묶음.
    predict/predict_proba에서 train과 동일한 변환을 적용하기 위해 selection 인덱스와
    poly transformer를 같이 들고 다님."""

    def __init__(self, model, top_indices=None, poly=None):
        self.model = model
        self.top_indices = top_indices
        self.poly = poly

    def transform(self, x):
        if self.top_indices is not None:
            x = x[:, self.top_indices]
        if self.poly is not None:
            x = self.poly.transform(x)
        return x


def _select_top_k(x, y, k, method='pearson'):
    return top_k_indices(x, y, k, method=method)


def train(x_train, y_train, x_valid=None, y_valid=None,
          n_steps=50, iter_per_step=100,
          top_k=None, use_polynomial=True, poly_degree=2,
          selection_method='pearson',
          penalty='l2', C=1.0, solver=None):
    # solver 자동 선택: penalty에 맞는 기본 솔버
    if solver is None:
        solver = 'saga' if penalty == 'l1' else 'lbfgs'

    stages = [f"raw {x_train.shape}"]

    top_indices = None
    if top_k is not None and top_k < x_train.shape[1]:
        top_indices = _select_top_k(x_train, y_train, top_k, method=selection_method)
        x_train = x_train[:, top_indices]
        if x_valid is not None:
            x_valid = x_valid[:, top_indices]
        stages.append(f"top-{top_k} {x_train.shape}")

    poly = None
    if use_polynomial:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
        x_train = poly.fit_transform(x_train)
        if x_valid is not None:
            x_valid = poly.transform(x_valid)
        stages.append(f"poly(d={poly_degree}) {x_train.shape}")

    print("  [LR] Pipeline:", " → ".join(stages))
    print(f"  [LR] Regularization: penalty={penalty}, C={C}, solver={solver}")

    model = _LR(max_iter=iter_per_step, warm_start=True,
                solver=solver, penalty=penalty, C=C)
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for _ in range(n_steps):
            model.fit(x_train, y_train)

            history["loss"].append(log_loss(y_train, model.predict_proba(x_train)))
            history["accuracy"].append(model.score(x_train, y_train))

            if x_valid is not None and y_valid is not None:
                history["val_loss"].append(log_loss(y_valid, model.predict_proba(x_valid)))
                history["val_accuracy"].append(model.score(x_valid, y_valid))

    return _LRPipeline(model, top_indices, poly), history


def predict(trained, x):
    return trained.model.predict(trained.transform(x))


def predict_proba(trained, x):
    return trained.model.predict_proba(trained.transform(x))[:, 1]
