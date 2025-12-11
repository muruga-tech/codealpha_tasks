
# SALES PREDICTION USING PYTHON 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# 1. LOAD DATA
data = pd.read_csv("sales_data.csv")
print("\n--- FIRST 5 ROWS OF DATA ---")
print(data.head())
if "Unnamed: 0" in data.columns:
    data = data.drop(columns=["Unnamed: 0"])
print("\nColumns found:", data.columns.tolist())
# 2. FEATURES & TARGET
# Input features 
X = data[["TV", "Radio", "Newspaper"]]

# Target variable
y = data["Sales"]
# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 4. PREPROCESSING + MODEL PIPELINE
numeric_features = ["TV", "Radio", "Newspaper"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features)
    ]
)
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)
# 5. TRAIN MODEL
model.fit(X_train, y_train)
# 6. PREDICTIONS
y_pred = model.predict(X_test)
# 7. EVALUATION
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n--- MODEL PERFORMANCE ---")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
# 8. PLOT: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()
# 9. PLOT: Advertising Impact (Correlation Heatmap)
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Advertising Spend & Sales")
plt.show()
# 10. INSIGHTS
print("\n--- BUSINESS INSIGHTS ---")
corr = data.corr()["Sales"].sort_values(ascending=False)
print("\nCorrelation with Sales:")
print(corr)
if corr["TV"] > corr["Radio"] and corr["TV"] > corr["Newspaper"]:
    print("\n• TV advertising has the strongest impact on sales.")
elif corr["Radio"] > corr["TV"]:
    print("\n• Radio advertising influences sales more than TV.")
else:
    print("\n• Newspaper ads have minimal impact on sales.")
print("• The regression model successfully predicts how ad spend affects sales.")
print("• This can guide companies in optimizing their marketing budget.")
print("\nAnalysis Completed Successfully!")