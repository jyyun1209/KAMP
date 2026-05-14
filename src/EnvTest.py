from importlib import import_module


# display_name : (import_path, pip_install_name)
_REQUIRED = {
    "numpy":       ("numpy",      "numpy"),
    "pandas":      ("pandas",     "pandas"),
    "matplotlib":  ("matplotlib", "matplotlib"),
    "scipy":       ("scipy",      "scipy"),
    "sklearn":     ("sklearn",    "scikit-learn"),
    "tensorflow":  ("tensorflow", "tensorflow"),
    "xgboost":     ("xgboost",    "xgboost"),
}


def env_test():
    line = "=" * 70
    sep = "-" * 70

    print()
    print(line)
    print("  Environment Check")
    print(sep)

    width = max(len(name) for name in _REQUIRED)
    missing = []
    for name, (module_path, pip_name) in _REQUIRED.items():
        try:
            mod = import_module(module_path)
            version = getattr(mod, "__version__", "(unknown)")
            print(f"  [OK]    {name:<{width}}  {version}")
        except ImportError:
            print(f"  [FAIL]  {name:<{width}}  not installed  ->  pip install {pip_name}")
            missing.append(name)

    print(sep)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  GPU    Available  ({len(gpus)} device(s))")
            for g in gpus:
                details = tf.config.experimental.get_device_details(g)
                name = details.get("device_name", g.name)
                cc = details.get("compute_capability")
                cc_str = f" (compute capability {cc[0]}.{cc[1]})" if cc else ""
                print(f"         {name}{cc_str}")
        else:
            print(f"  GPU        Not Available  (CPU mode)")
    except ImportError:
        print(f"  GPU              n/a  (tensorflow not installed)")

    print(line)
    print()

    if missing:
        raise RuntimeError(f"Required libraries missing: {', '.join(missing)}")
