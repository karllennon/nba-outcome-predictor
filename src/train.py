import pandas as pd

df = pd.read_csv("/Users/karlgodel/Downloads/nba_games.csv")
print(df.head())


# Drop Missing Values
df.dropna(inplace=True)

#Encode 'Win" colum as 1 or 0
df["Win"] = df["PTS_home"] > df["PTS_away"]
df["Win"] = df["Win"].astype(int)

# Select fearures
features = ["FG_PCT_home", "FG_PCT_away", "FT_PCT_home", "FT_PCT_away", 
            "AST_home", "AST_away", "REB_home", "REB_away"]
X = df[features]
y = df["Win"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

#Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

#Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)

#Example prediction
new_game = pd.DataFrame([{
    "FG_PCT_home": 0.18,
    "FG_PCT_away": 0.06,
    "FT_PCT_home": 0.80,
    "FT_PCT_away": 0.05,
    "AST_home": 8,
    "AST_away": 12,
    "REB_home": 90,
    "REB_away": 105
}])

prediction = model.predict(new_game)    
print(f"Predicted outcome (1 for home win, 0 for away win): {prediction[0]}")

# Save metrics, plot, and model

import os, json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

# Make sure output folders exist 
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Get predicted probabilities for the positive class (home win)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate common classigication metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob))
}
print("Metrics:", metrics)

# Save the metrics to a JSON format for later reference
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Create adn save the confusion matrix plot
disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/figures/confusion_matrix.png")
plt.close()

# Save the trained model using joblib
import joblib
joblib.dump(model, "models/nba_game_outcome_model.joblib")

# Let the user know where the files are saved
print("Metrics saved to reports/metrics.json")
print("Confusion matrix plot saved to reports/figures/confusion_matrix.png")
print("Model saved to models/nba_game_outcome_model.joblib")
print("All files saved successfully.")
print("You can now use the model to predict game outcomes.")