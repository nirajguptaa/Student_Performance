import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. DATASET
df = pd.DataFrame({
    "hours_studied": [1,2,3,4,5,6,7,8,9,10],
    "previous_score": [40,45,50,55,60,65,70,75,80,85],
    "pass_fail": [0,0,0,0,1,1,1,1,1,1]
})

X = df[["hours_studied", "previous_score"]]
y = df["pass_fail"]

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. MODELS
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 5. EVALUATION
print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n--- Random Forest ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# 6. SAVE BEST MODEL
joblib.dump(rf, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model & Scaler Saved")