from sklearn.model_selection import train_test_split


def train_val_split(data, val_ratio=0.2, stratify=True, shuffle=True, random_state=None):
    x, y = data

    if stratify and shuffle:
        strat = y
    else:
        strat = None

    x_train, x_val, y_train, y_val = \
        train_test_split(
            x, y, test_size=val_ratio,
            stratify=strat, shuffle=shuffle, random_state=random_state,
        )

    return (x_train, y_train), (x_val, y_val)
