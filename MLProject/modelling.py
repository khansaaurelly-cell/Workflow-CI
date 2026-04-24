import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import numpy as np

# DEBUG
print("CURRENT DIR:", os.getcwd())
print("FILES:", os.listdir())


os.chdir(os.path.dirname(__file__))
print("FIXED DIR:", os.getcwd())

# =========================
# LOAD DATA
# =========================
file_path = "titanic_preprocessing.csv"
df = pd.read_csv(file_path)

for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].astype(float)

# =========================
# SPLIT DATA
# =========================
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SET MLFLOW
# =========================
os.chdir(os.path.dirname(__file__))
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("titanic_model")

mlflow.autolog()

# =========================
# TRAINING
# =========================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print("Accuracy:", acc)

    mlflow.log_metric("accuracy_manual", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)