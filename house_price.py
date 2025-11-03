import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
data = pd.DataFrame({
    "Size": [1000, 1500, 2000, 2500, 3000, 3500, 4000],
    "Bedrooms": [2, 3, 3, 4, 4, 5, 5],
    "Price": [50, 70, 90, 110, 130, 150, 170] 
})
X = data[["Size","Bedrooms"]]
y = data["Price"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y, test_size = 0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
print("\nPredicted prices on test data:",pred)
print(f"MSE: {mean_squared_error(y_test, pred):.2f}")
size = float(input("🏠 Enter house size (in sqft): "))
bedrooms = int(input("🛏️ Enter number of bedrooms: "))
new_data = pd.DataFrame([[size, bedrooms]],columns=["Size","Bedrooms"])
new_data_scaled = scaler.transform(new_data)
predicted_price = model.predict(new_data_scaled)
print(f"💰 Predicted Price (in lakhs): {predicted_price[0]:.2f}\n")
fig = plt.figure()
plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(data["Size"], data["Bedrooms"], data["Price"], color='blue', label='Actual Data')
ax.scatter([size],[bedrooms],[predicted_price],color='red', s=100, label='Predicted Point')
ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price (lakhs)")
ax.set_title("3D House Price Prediction", fontsize=14, pad=20)
ax.view_init(elev=35, azim=120) 
ax.legend()
plt.tight_layout()
plt.show()
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")