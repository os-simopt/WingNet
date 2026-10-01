# PGPE Standalone Optimizer

This module provides a minimal, framework-agnostic implementation of PGPE
(Policy Gradient with Parameter-based Exploration) for hyperparameter optimization.

You define:

- a search space via `ParameterSpec`,
- an `objective_fn(params)` that builds/trains/evaluates your model,

and call `pgpe_optimize(...)` to obtain the best parameter configuration.

---

## Example (Keras)

```python
from pgpe_standalone import ParameterSpec, pgpe_optimize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Search space
search_space = [
    ParameterSpec("learning_rate", 1e-5, 1e-2, init=1e-3),
    ParameterSpec("num_units", 16, 512, init=128, is_int=True, sigma_init=50),
    ParameterSpec("dropout", 0.0, 0.7, init=0.3),
]

def build_model(input_shape, params: dict) -> keras.Model:
    lr = params["learning_rate"]
    num_units = int(params["num_units"])
    dropout = params["dropout"]

    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(num_units, activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(num_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)  # regression example

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mse"],
    )
    return model

# Example data (replace with your own)
x_train, y_train = ...
x_val, y_val = ...
input_shape = x_train.shape[1:]

def objective(params: dict) -> float:
    model = build_model(input_shape, params)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=0,
    )
    val_mse = history.history["val_mse"][-1]
    # PGPE maximizes reward -> use negative loss
    return -float(val_mse)

best_params, hist = pgpe_optimize(
    search_space=search_space,
    objective_fn=objective,
    n_iterations=30,
    population_size=8,
    lr_mu=0.1,
    lr_sigma=0.05,
    seed=42,
)

print("Best parameters found:", best_params)
```


Using PGPE with other models (PyTorch, WingNet, etc.)
For any other model (e.g. PyTorch, WingNet), implement:

```python
def objective(params: dict) -> float:
    # 1) Inject params into your config
    # 2) Run a short training loop
    # 3) Evaluate on a validation set
    # 4) Return a scalar reward (e.g. val_accuracy or macro-F1)
    return reward
```

Then call pgpe_optimize(...) with the same search space.