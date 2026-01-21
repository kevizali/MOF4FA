import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Load & preprocess
df = pd.read_csv("#dataset_location")

# Feature and target
feature_df = df.iloc[:, :-2].copy()
X = feature_df
y = df[["target_feature"]]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "SVR": SVR(kernel='rbf', C=200, epsilon=0.1),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(verbosity=0, random_state=42),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
}

# Train and evaluate
print("Model Performance (R², RMSE, MAE, SRCC, CV R²):\n")
results = {}

for model_name, model in models.items():
    print(f"--- {model_name} ---")

    for col in y.columns:
        # Train model
        model.fit(X_train_scaled, y_train[col])
        y_pred = model.predict(X_test_scaled)

        # Inverse transform FA_select_log
        if col == "FA_select_log":
            y_pred = np.expm1(y_pred)
            y_true = np.expm1(y_test[col])
            label = "FA_select"
        else:
            y_true = y_test[col]
            label = col

        # R² Score
        r2 = r2_score(y_true, y_pred)

        # RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # MAE
        mae = mean_absolute_error(y_true, y_pred)

        # SRCC
        srcc, _ = spearmanr(y_true, y_pred)

        print(f"{label}: R² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, SRCC = {srcc:.4f}")

        # Cross-validation (CV = 5)
        cv_scores = cross_val_score(model, X_train_scaled, y_train[col], cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        print(f"{label}: CV R² (mean) = {cv_mean:.4f}")

        results[(model_name, label)] = (r2, rmse, mae, srcc, cv_mean)

    print()