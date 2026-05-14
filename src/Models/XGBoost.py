from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

parameters = {
    "learning_rate" : [0.1, 0.01, 0.001, 0.0001],
    # "eta" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "max_depth" : [3, 4, 5, 6, 8, 10, 12, 15],
    "min_child_weight" : [1, 3, 5, 7],
    "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
    "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
}

def train(x_train, y_train, x_valid=None, y_valid=None, withGridSearch=False):
    model = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=3,
        gamma=0.2,
        subsample=0.6,
        colsample_bytree=1.0,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        eval_metric=['auc', 'error', 'logloss'],
        seed=0)
    
    if (withGridSearch):
        parameters = {
            "learning_rate" : [0.1, 0.01, 0.001, 0.0001],
            # "eta" : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth" : [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight" : [1, 3, 5, 7],
            "gamma" : [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree" : [0.3, 0.4, 0.5, 0.7]
        }

        grid = GridSearchCV(model,
                            parameters, n_jobs=4,
                            scoring="neg_log_loss",
                            cv=5)
        
        grid.fit(
            x_train.squeeze(),
            y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            verbose=False)
        model = grid.best_estimator_

    else:
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_valid, y_valid)],
            verbose=False
        )
    
    history = _to_history(model.evals_result())
    return model, history


def _to_history(evals_result):
    train_m = evals_result.get("validation_0", {})
    valid_m = evals_result.get("validation_1", {})
    history = {}
    if "error" in train_m:
        history["accuracy"] = [1 - e for e in train_m["error"]]
    if "error" in valid_m:
        history["val_accuracy"] = [1 - e for e in valid_m["error"]]
    if "logloss" in train_m:
        history["loss"] = list(train_m["logloss"])
    if "logloss" in valid_m:
        history["val_loss"] = list(valid_m["logloss"])
    return history if history else None


def predict(model, x):
    return model.predict(x)


def predict_proba(model, x):
    return model.predict_proba(x)[:, 1]
