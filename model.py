import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("sample.csv")  # or "train.csv" if you downloaded Kaggle one

# Show correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[["GrLivArea", "BedroomAbvGr", "FullBath", "SalePrice"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("📊 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Select features & target
X = df[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = df["SalePrice"]

# Check missing values
print("\n🔍 Missing values:\n", X.isnull().sum())

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"📈 R² Score: {r2:.4f}")
print(f"🧮 Mean Squared Error: {mse:,.2f}")

# Plot: Actual vs Predicted with y = x line
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel("Actual Sale Price")
plt.ylabel("Predicted Sale Price")
plt.title("🎯 Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True, color='darkgreen')
plt.title("🧪 Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()





