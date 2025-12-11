
# Car Price Prediction 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
data = pd.read_csv("car_data.csv")

print("\n--- FIRST 5 ROWS OF DATA ---")
print(data.head())

# 2. Clean missing values
data = data.dropna()

# 3. Select input features and target
X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# Identify categorical & numerical columns
categorical_cols = ["Car_Name", "Fuel_Type", "Selling_type", "Transmission"]
numerical_cols = ["Year", "Present_Price", "Driven_kms", "Owner"]

# 4. Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 5. Model pipeline with RandomForest
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=250, random_state=42))
])

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train the model
model.fit(X_train, y_train)

# 8. Predict selling prices
y_pred = model.predict(X_test)

# 9. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- MODEL PERFORMANCE ---")
print(f"MAE  (Mean Absolute Error): {mae:.2f}")
print(f"MSE  (Mean Squared Error): {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# 10. Actual vs Predicted plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Car Selling Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.show()

# 11. Feature Importance (from RandomForest)
regressor = model.named_steps["regressor"]

# Get feature names after encoding
cat_features = model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
num_features = numerical_cols
all_features = np.concatenate((num_features, cat_features))

importances = regressor.feature_importances_

# Sort features
sorted_idx = np.argsort(importances)

plt.figure(figsize=(12,8))
plt.barh(all_features[sorted_idx], importances[sorted_idx])
plt.title("Feature Importance in Car Price Prediction")
plt.xlabel("Importance Score")
plt.show()

print("\nAnalysis Completed Successfully!")


