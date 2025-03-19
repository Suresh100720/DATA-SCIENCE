import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
sales_df = pd.read_csv("advertising.csv", encoding="ISO-8859-1")
sales_df_cleaned = sales_df.drop(columns=["Unnamed: 0"], errors="ignore")
X = sales_df_cleaned.drop(columns=["Sales"]) 
y = sales_df_cleaned["Sales"]  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Sales Prediction Model MAE: {mae:.2f}")
def predict_sales():
    print("\nEnter the advertising expenditure for:")
    try:
        tv = float(input("TV Advertising ($): "))
        radio = float(input("Radio Advertising ($): "))
        newspaper = float(input("Newspaper Advertising ($): "))
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return
    input_data = pd.DataFrame([[tv, radio, newspaper]], columns=X.columns)
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=X.columns)
    predicted_sales = model.predict(input_scaled)[0]
    print(f"\nPredicted Sales: ${predicted_sales:.2f}")
predict_sales()
