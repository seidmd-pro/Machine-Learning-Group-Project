
 #Import ML classification pipeline and evaluation metrics
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
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)


# PART 1. CREATE DATASET (30 RECORDS - HIGH OVERLAP)


data = [
    # ID, Temperature_C, Vibration_mm_s, Speed_RPM, Operating_Hours, Failure_Risk
    [1,  55, 2.1, 1480, 1500, 0],
    [2,  62, 2.8, 1475, 2100, 0],
    [3,  58, 2.4, 1460, 1800, 0],
    [4,  72, 3.8, 1420, 3600, 0], 
    [5,  68, 3.2, 1440, 2900, 0],
    [6,  59, 2.5, 1455, 1950, 0],
    [7,  65, 3.0, 1450, 2600, 0],
    [8,  60, 2.7, 1465, 2200, 0],
    [9,  74, 4.2, 1410, 3800, 0], 
    [10, 56, 2.2, 1470, 1700, 0],
    [11, 69, 3.5, 1430, 3200, 0],
    [12, 64, 2.9, 1445, 2400, 0],
    [13, 67, 3.1, 1435, 2800, 0],
    [14, 57, 2.3, 1468, 1600, 0],
    [15, 63, 2.8, 1452, 2300, 0],
    [16, 75, 4.8, 1380, 4200, 1],
    [17, 85, 6.0, 1340, 5500, 1],
    [18, 92, 6.8, 1310, 6200, 1],
    [19, 71, 4.1, 1410, 3900, 1], 
    [20, 88, 6.3, 1320, 5800, 1],
    [21, 78, 5.2, 1370, 4600, 1],
    [22, 95, 7.0, 1300, 6500, 1],
    [23, 73, 4.5, 1400, 4100, 1],
    [24, 82, 5.7, 1350, 5100, 1],
    [25, 86, 6.1, 1335, 5600, 1],
    [26, 90, 6.6, 1315, 6000, 1],
    [27, 77, 5.0, 1375, 4400, 1],
    [28, 89, 6.4, 1325, 5900, 1],
    [29, 80, 5.4, 1360, 4800, 1],
    [30, 94, 6.9, 1305, 6300, 1],
]


columns = [
    "Machine_ID", "Temperature_C", "Vibration_mm_s",
    "Speed_RPM", "Operating_Hours", "Failure_Risk"
]
df = pd.DataFrame(data, columns=columns)

print("\nDATASET: 30 MACHINE RECORDS (HIGH OVERLAP)")
print(df.to_string(index=False))
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

# Separate predictors (X) and target (y).
feature_cols = ["Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"]
X = df[feature_cols]
y = df["Failure_Risk"]

# Initial data visualization.
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", s=100
)
plt.title("Machine temperature vs. vibration (High Overlap)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[feature_cols + ["Failure_Risk"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation matrix of engineering variables")
plt.tight_layout()
plt.show()
