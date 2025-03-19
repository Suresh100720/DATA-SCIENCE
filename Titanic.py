import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
titanic_df = pd.read_csv("tested.csv", encoding="ISO-8859-1")
titanic_df_cleaned = titanic_df.drop(columns=["Name", "Ticket", "Cabin"])
titanic_df_cleaned["Age"].fillna(titanic_df_cleaned["Age"].median(), inplace=True)
titanic_df_cleaned["Fare"].fillna(titanic_df_cleaned["Fare"].median(), inplace=True)
titanic_df_cleaned["Embarked"].fillna(titanic_df_cleaned["Embarked"].mode()[0], inplace=True)
label_encoders = {}
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    titanic_df_cleaned[col] = le.fit_transform(titanic_df_cleaned[col])
    label_encoders[col] = le
X = titanic_df_cleaned.drop(columns=["Survived", "PassengerId"])
y = titanic_df_cleaned["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Titanic Survival Prediction Model Accuracy: {accuracy:.2f}")
def predict_survival_by_id():
    try:
        passenger_id = int(input("Enter Passenger ID: "))
    except ValueError:
        print("Invalid input! Please enter a valid Passenger ID.")
        return
    passenger_data = titanic_df_cleaned[titanic_df_cleaned["PassengerId"] == passenger_id]
    if passenger_data.empty:
        print(f"Passenger ID {passenger_id} not found!")
        return
    passenger_features = passenger_data.drop(columns=["Survived", "PassengerId"])
    prediction = log_reg.predict(passenger_features)[0]
    actual = "Survived" if passenger_data["Survived"].values[0] == 1 else "Did not Survive"
    predicted = "Survived" if prediction == 1 else "Did not Survive"
    print(f"Passenger ID {passenger_id} -> Actual: {actual}, Predicted: {predicted}")
predict_survival_by_id()
