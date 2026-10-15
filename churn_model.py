
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load and clean the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)
df = df.dropna(subset=["TotalCharges"])

# 2. Prepare features and target
# Remove customerID because it is an identifier, not a useful
# customer-behaviour feature.
X = df.drop(columns=["Churn", "customerID"])
y = (df["Churn"] == "Yes").astype(int)

# 3. Identify numeric and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_columns),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ]
)

# 4. Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Build and train the model
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

model.fit(X_train, y_train)

# 6. Make predictions on unseen test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# 7. Evaluate the model
metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(
        y_test, predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, predictions, zero_division=0
    ),
    "F1-score": f1_score(
        y_test, predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, probabilities)
}

print("CUSTOMER CHURN MODEL EVALUATION")
print("=" * 40)

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification report:")
report = classification_report(
    y_test,
    predictions,
    target_names=["Stayed", "Churned"],
    zero_division=0
)
print(report)

# Save results for the project portfolio
with open("model_results.txt", "w") as file:
    file.write("CUSTOMER CHURN MODEL EVALUATION\n\n")
    for name, value in metrics.items():
        file.write(f"{name}: {value:.4f}\n")
    file.write("\nClassification report:\n")
    file.write(report)

# 8. Save confusion matrix chart
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Stayed", "Churned"],
    cmap="Blues"
)
plt.title("Customer Churn Model - Confusion Matrix")
plt.tight_layout()
plt.savefig("churn_confusion_matrix.png", dpi=150)
plt.close()

print("\nModel results and confusion matrix saved.")

