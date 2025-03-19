import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
file_path = "IRIS.csv"  
iris_df = pd.read_csv(file_path)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
label_encoder = LabelEncoder()
iris_df["species"] = label_encoder.fit_transform(iris_df["species"])
X = iris_df.drop(columns=["species"])  
y = iris_df["species"]  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nIris Classification Model Accuracy: {accuracy:.2f}")
def predict_iris_species():
    try:
        sepal_length = float(input("\nEnter Sepal Length (cm): "))
        sepal_width = float(input("Enter Sepal Width (cm): "))
        petal_length = float(input("Enter Petal Length (cm): "))
        petal_width = float(input("Enter Petal Width (cm): "))
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                  columns=X.columns)
        input_scaled = scaler.transform(input_data)
        input_scaled_df = pd.DataFrame(input_scaled, columns=X.columns)
        species_index = model.predict(input_scaled_df)[0]
        predicted_species = label_encoder.inverse_transform([species_index])[0]
        print(f"\nPredicted Iris Species: {predicted_species}")
    except Exception as e:
        print(f"Error: {e}")
predict_iris_species()
