# Import ML classification pipeline and evaluation metrics
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

# PART 1. CREATE DATASET (30 PUMP RECORDS - 3 CLASSES)
# Input features: Vibration (mm/s), RPM, temperature (°C), pressure, operating hours[cite: 5].

data = [
    # Normal Pumps (0)
    [1, 2.0, 1490, 52, 120, 1500, 0],
    [2, 2.3, 1485, 55, 118, 1750, 0],
    [3, 1.9, 1500, 50, 122, 1200, 0],
    [4, 2.5, 1475, 58, 119, 2100, 0],
    [5, 2.1, 1480, 54, 121, 1600, 0],
    [6, 2.4, 1470, 59, 117, 1950, 0],
    [7, 1.8, 1495, 51, 123, 1300, 0],
    [8, 2.6, 1465, 60, 116, 2200, 0],
    [9, 2.2, 1488, 53, 120, 1450, 0],
    [10, 2.7, 1460, 61, 115, 2400, 0],

    # Warning Pumps (1) - Showing early signs of degradation
    [11, 4.5, 1430, 72, 108, 3500, 1],
    [12, 4.1, 1445, 68, 112, 3100, 1],
    [13, 4.8, 1410, 76, 105, 4100, 1],
    [14, 4.3, 1435, 70, 110, 3300, 1],
    [15, 4.6, 1420, 74, 107, 3800, 1],
    [16, 4.0, 1450, 67, 113, 2900, 1],
    [17, 4.9, 1405, 77, 104, 4300, 1],
    [18, 4.2, 1440, 69, 111, 3200, 1],
    [19, 4.7, 1415, 75, 106, 3950, 1],
    [20, 4.4, 1425, 71, 109, 3600, 1],

    # Fault Pumps (2) - Critical failure conditions
    [21, 7.2, 1310, 92, 85, 6200, 2],
    [22, 6.8, 1330, 88, 90, 5600, 2],
    [23, 7.5, 1290, 95, 80, 6800, 2],
    [24, 6.5, 1350, 85, 93, 5100, 2],
    [25, 7.8, 1270, 98, 76, 7100, 2],
    [26, 7.0, 1320, 90, 87, 5900, 2],
    [27, 8.1, 1250, 102, 72, 7500, 2],
    [28, 6.7, 1340, 87, 91, 5400, 2],
    [29, 7.4, 1300, 94, 82, 6500, 2],
    [30, 6.9, 1325, 89, 88, 5800, 2],
]

columns = [
    "Pump_ID", "Vibration_mm_s", "Speed_RPM",
    "Temperature_C", "Pressure_psi", "Operating_Hours", "Pump_Condition"
]
df = pd.DataFrame(data, columns=columns)
target_names = ["Normal", "Warning", "Fault"]

print("\nDATASET: 30 PUMP RECORDS")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Pump_Condition"].value_counts().rename(index={0: "Normal", 1: "Warning", 2: "Fault"}))

# Separate predictors (X) and target (y).
feature_cols = ["Vibration_mm_s", "Speed_RPM", "Temperature_C", "Pressure_psi", "Operating_Hours"]
X = df[feature_cols]
y = df["Pump_Condition"]

# Initial data visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Vibration_mm_s", y="Pressure_psi",
    hue="Pump_Condition", palette="deep", s=100
)
plt.title("Pump Vibration vs. Pressure (3 Classes)")
plt.xlabel("Vibration (mm/s)")
plt.ylabel("Pressure (psi)")
plt.legend(title="0=Normal, 1=Warning, 2=Fault")
plt.tight_layout()
plt.show()

# PART 3. DEFINE FIVE INDEPENDENT CLASSIFIERS
# ============================================================
# Potential algorithms from case study[cite: 5]

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)

random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)

adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2, random_state=42),
    n_estimators=50,
    learning_rate=0.5,
    random_state=42
)

models = {
    "SVM": svm_model,
    "Naïve Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}

# Keep the class balance in both subsets using stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# PART 4. TRAIN, PREDICT, EVALUATE, AND VISUALIZE EACH MODEL
# Evaluation priority: Recall for Fault, precision for Fault, macro F1, confusion matrix[cite: 5].

results = []
predictions = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(f"{model_name.upper()}")
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    # Calculate overall and Fault-specific evaluation metrics[cite: 5].
    # '2' is the label for Fault.
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    fault_precision = precision_score(y_test, y_pred, labels=[2], average=None, zero_division=0)[0]
    fault_recall = recall_score(y_test, y_pred, labels=[2], average=None, zero_division=0)[0]

    results.append({
        "Algorithm": model_name,
        "Macro F1": macro_f1,
        "Fault Precision": fault_precision,
        "Fault Recall": fault_recall
    })

    print(f"Priority Metrics:")
    print(f"Macro F1-score      : {macro_f1:.3f}")
    print(f"Precision for Fault : {fault_precision:.3f}")
    print(f"Recall for Fault    : {fault_recall:.3f}")
    
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    ))

    # Confusion matrix visualization[cite: 5]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=target_names,
        cmap="Blues",
        colorbar=False,
        ax=ax
    )
    ax.set_title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()


# PART 5. COMPARE THE FIVE ALGORITHMS
# ============================================================

results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON (PRIORITY METRICS)")
print(results_df.round(3))

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of supervised algorithms (Pump Condition)")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
# ============================================================
# PART 6. INTERPRET TREE-BASED FEATURE IMPORTANCE
# ============================================================

tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}
for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    print(f"\n{name} feature importance:")
    print(importance.round(3))

    plt.figure(figsize=(7, 4))
    plt.barh(importance.index, importance.values)
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Pump Feature")
    plt.tight_layout()
    plt.show()