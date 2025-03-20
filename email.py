import pandas as pd
import numpy as np
import joblib  # Salvare și încărcare model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Funcție pentru încărcarea datelor
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

# 2. Funcție pentru preprocesarea datelor
def preprocess_data(df, is_train=True):
    df = df.copy()  # Evităm modificarea directă a originalului
    if "uuid" in df.columns:
        df = df.drop(columns=["uuid"])  # Eliminăm identificatorul unic
    if is_train and "label" in df.columns:
        y = df["label"]
        X = df.drop(columns=["label"])
        return X, y
    return df  # În cazul testului, returnăm doar X

# 3. Încărcare și preprocesare date
train_df, test_df = load_data("train_data.csv", "test_data.csv")
X_train, y_train = preprocess_data(train_df)
X_test = preprocess_data(test_df, is_train=False)

# 4. Împărțire în set de antrenament și test pentru validare
X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 5. Antrenare model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_split, y_train_split)

# 6. Evaluare model
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Accuratețea modelului: {accuracy:.2f}")

# 7. Salvarea modelului
joblib.dump(model, "model_rf.pkl")

# 8. Generarea predicțiilor finale
final_predictions = model.predict(X_test)

# 9. Salvarea predicțiilor într-un fișier CSV
test_df["predicted_label"] = final_predictions
test_df.to_csv("predictions.csv", index=False)

print("Predicțiile au fost salvate în 'predictions.csv'!")
