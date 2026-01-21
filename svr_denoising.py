import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

# Load and preprocess the dataset
df = pd.read_csv("#dataset_path")
df = df.drop(columns=df.columns[0])

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Train a SVM model
rf = SVR(kernel='rbf', C=10, epsilon=0.1)

rf.fit(X_scaled, y)

# Predict FA values (denoised values)
y_pred = rf.predict(X_scaled)

# Blend original and predicted values for denoising
alpha = 0.9
y_denoised = alpha * y_pred + (1 - alpha) * y


# Save to new CSV
df_denoised = df.copy()
df_denoised["FA_denoised"] = y_denoised
df_denoised.to_csv("denoised_data.csv", index=False)