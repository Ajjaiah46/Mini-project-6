import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location_score"]


def generate_sample_data(n=1000, seed=42):
    """Create synthetic house data. Price is in Rs. lakhs."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "area_sqft": rng.integers(500, 4000, n),
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 5, n),
        "age_years": rng.integers(0, 40, n),
        "location_score": rng.integers(1, 11, n),  # 1 = poor, 10 = prime
    })
    noise = rng.normal(0, 8, n)
    df["price_lakhs"] = (
        0.045 * df["area_sqft"]
        + 5 * df["bedrooms"]
        + 4 * df["bathrooms"]
        - 0.6 * df["age_years"]
        + 6 * df["location_score"]
        + noise
    )
    return df


def build_model():
    # Neural networks need scaled inputs, so scaler + MLP are bundled in one pipeline.
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),   # two hidden layers
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,           # stops when validation score stops improving
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=42,
        ),
    )


def train():
    df = generate_sample_data()
    X, y = df[FEATURES], df["price_lakhs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print("MAE :", round(mean_absolute_error(y_test, pred), 2), "lakhs")
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, pred)), 2), "lakhs")
    print("R2  :", round(r2_score(y_test, pred), 4))

    mlp = model[-1]
    print(f"\nTrained for {mlp.n_iter_} epochs, layers: {[w.shape for w in mlp.coefs_]}")

    joblib.dump(model, "house_model.pkl")
    df.to_csv("house_data.csv", index=False)
    print("Saved house_model.pkl and house_data.csv")


if __name__ == "__main__":
    train()
