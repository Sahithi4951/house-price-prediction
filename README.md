# House Price Prediction using Multiple Linear Regression

## Overview  
This project predicts **house prices** based on two main features:
- Size (in sqft)  
- Number of Bedrooms

It uses **Multiple Linear Regression** from `scikit-learn` to build a predictive model.  
The project also includes **3D visualization** with `matplotlib`, and saves the trained model using `joblib` for future use.

---

## Features  
- Predict house prices using Linear Regression  
- 3D data visualization  
- Feature scaling with `StandardScaler`  
- Model saving and reloading with `joblib`  
- Beginner-friendly, clean Python code  

---

## Tech Stack
**Language:** Python 3.x  
**Libraries:**  
- pandas  
- numpy  
- matplotlib  
- scikit-learn  
- joblib  

---

## How It Works  

1. The dataset contains house `Size`, `Bedrooms`, and `Price`.  
2. Features are scaled using `StandardScaler`.  
3. The model is trained using `LinearRegression`.  
4. The user can input custom values for size and bedrooms to get a prediction.  
5. The predicted price is shown both in the console and in a 3D plot.

---

## Example Output

Predicted prices on test data: [50. 70.]
MSE: 0.0
Enter house size: 2750
Enter the number of bedrooms: 4
Predicted Price (in lakhs): [120.]
---

## Visualization

The 3D scatter plot shows:
- Blue dots → Actual data points  
- Red dot → Your predicted house price  

---

## Future Improvements
- Add more features like location, age of house, etc.  
- Use a larger real-world dataset  
- Build a Flask / Streamlit web app for live predictions  

---

## Author
**Sahithi Bashetty**  
bashettysahithi@gmail.com  
Built with Python and scikit-learn
