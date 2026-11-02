from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso

)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

data = pd.read_csv("data/student_data.csv")

print("\n===================================")
print("       SCORESENSE PROJECT")
print("===================================")

print("\nDataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Sleep_Hours"
]

X = data[features]

# Regression target
y = data["Final_Score"]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_predictions
    )
)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print("\n===== LINEAR REGRESSION =====")

print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 3))


# ==========================================
# 5. POLYNOMIAL REGRESSION
# ==========================================

polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

polynomial_model.fit(
    X_train,
    y_train
)

polynomial_predictions = polynomial_model.predict(
    X_test
)

polynomial_mae = mean_absolute_error(
    y_test,
    polynomial_predictions
)

polynomial_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        polynomial_predictions
    )
)

polynomial_r2 = r2_score(
    y_test,
    polynomial_predictions
)

print("\n===== POLYNOMIAL REGRESSION =====")

print("MAE:", round(polynomial_mae, 2))
print("RMSE:", round(polynomial_rmse, 2))
print("R2 Score:", round(polynomial_r2, 3))


# ==========================================
# 6. RIDGE REGRESSION
# ==========================================

ridge_model = Ridge(alpha=1.0)

ridge_model.fit(
    X_train,
    y_train
)

ridge_predictions = ridge_model.predict(
    X_test
)

ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_predictions
    )
)

ridge_r2 = r2_score(
    y_test,
    ridge_predictions
)

print("\n===== RIDGE REGRESSION =====")

print("MAE:", round(ridge_mae, 2))
print("RMSE:", round(ridge_rmse, 2))
print("R2 Score:", round(ridge_r2, 3))


# ==========================================
# 7. LASSO REGRESSION
# ==========================================

lasso_model = Lasso(alpha=0.1)

lasso_model.fit(
    X_train,
    y_train
)

lasso_predictions = lasso_model.predict(
    X_test
)

lasso_mae = mean_absolute_error(
    y_test,
    lasso_predictions
)

lasso_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lasso_predictions
    )
)

lasso_r2 = r2_score(
    y_test,
    lasso_predictions
)

print("\n===== LASSO REGRESSION =====")

print("MAE:", round(lasso_mae, 2))
print("RMSE:", round(lasso_rmse, 2))
print("R2 Score:", round(lasso_r2, 3))


# ==========================================
# 8. MODEL COMPARISON
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "MAE": [
        linear_mae,
        polynomial_mae,
        ridge_mae,
        lasso_mae
    ],

    "RMSE": [
        linear_rmse,
        polynomial_rmse,
        ridge_rmse,
        lasso_rmse
    ],

    "R2 Score": [
        linear_r2,
        polynomial_r2,
        ridge_r2,
        lasso_r2
    ]
})


print("\n===================================")
print("       REGRESSION COMPARISON")
print("===================================")

print(
    results.round(3).to_string(index=False)
)


# ==========================================
# 9. GRAPH
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    results["Model"],
    results["R2 Score"]
)

plt.title("Regression Model Comparison")

plt.xlabel("Model")

plt.ylabel("R² Score")

plt.xticks(rotation=20)

plt.tight_layout()

plt.show()

# ==========================================
# 10. CLASSIFICATION DATA
# ==========================================

X_classification = data[features]

y_classification = data["Result"]


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(

    X_classification,

    y_classification,

    test_size=0.2,

    random_state=42,

    stratify=y_classification
)


# ==========================================
# 11. ENCODE TARGET
# ==========================================

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(
    y_train_cls
)

y_test_encoded = encoder.transform(
    y_test_cls
)


# ==========================================
# 12. LOGISTIC REGRESSION
# ==========================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(

    X_train_cls,

    y_train_encoded
)

logistic_predictions = logistic_model.predict(
    X_test_cls
)

logistic_accuracy = accuracy_score(

    y_test_encoded,

    logistic_predictions
)


print("\n===================================")
print("       LOGISTIC REGRESSION")
print("===================================")

print(
    "Accuracy:",
    round(logistic_accuracy, 3)
)


# ==========================================
# 13. SCALE DATA FOR KNN
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_cls
)

X_test_scaled = scaler.transform(
    X_test_cls
)


# ==========================================
# 14. KNN
# ==========================================

knn_model = KNeighborsClassifier(
    n_neighbors=5
)

knn_model.fit(

    X_train_scaled,

    y_train_encoded
)

knn_predictions = knn_model.predict(
    X_test_scaled
)

knn_accuracy = accuracy_score(

    y_test_encoded,

    knn_predictions
)


print("\n===================================")
print("              KNN")
print("===================================")

print(
    "Accuracy:",
    round(knn_accuracy, 3)
)


# ==========================================
# 15. CLASSIFICATION COMPARISON
# ==========================================

classification_results = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "KNN"
    ],

    "Accuracy": [
        logistic_accuracy,
        knn_accuracy
    ]
})


print("\n===================================")
print("    CLASSIFICATION COMPARISON")
print("===================================")

print(
    classification_results.round(3).to_string(
        index=False
    )
)
# ==========================================
# 16. CLASSIFICATION MODEL GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    classification_results["Model"],
    classification_results["Accuracy"]
)

plt.title("Classification Model Comparison")

plt.xlabel("Model")

plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.show()
# ==========================================
# 17. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test_encoded,
    logistic_predictions
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.xticks(
    range(len(encoder.classes_)),
    encoder.classes_
)

plt.yticks(
    range(len(encoder.classes_)),
    encoder.classes_
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()

# ==========================================
# 18. STUDENT PREDICTION DEMO
# ==========================================

print("\n")
print("===================================")
print("       STUDENT PREDICTION")
print("===================================")

print("\nEnter student details:")

study_hours = float(input("Study Hours (1-8): "))

attendance = float(input("Attendance % (50-100): "))

previous_score = float(input("Previous Score (0-100): "))

assignments = int(input("Assignments Completed (0-10): "))

sleep_hours = float(input("Sleep Hours (4-9): "))


# Create input dataframe

student = pd.DataFrame({

    "Study_Hours": [study_hours],

    "Attendance": [attendance],

    "Previous_Score": [previous_score],

    "Assignments_Completed": [assignments],

    "Sleep_Hours": [sleep_hours]

})


# ==========================================
# FINAL SCORE PREDICTION
# ==========================================

predicted_score = linear_model.predict(
    student
)[0]


# Keep score between 0 and 100

predicted_score = np.clip(
    predicted_score,
    0,
    100
)


# ==========================================
# LOGISTIC REGRESSION PREDICTION
# ==========================================

student_logistic = logistic_model.predict(
    student
)[0]

logistic_result = encoder.inverse_transform(
    [student_logistic]
)[0]


# ==========================================
# KNN PREDICTION
# ==========================================

student_scaled = scaler.transform(
    student
)

student_knn = knn_model.predict(
    student_scaled
)[0]

knn_result = encoder.inverse_transform(
    [student_knn]
)[0]


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n")
print("╔══════════════════════════════════════╗")
print("║          SCORESENSE RESULT           ║")
print("╠══════════════════════════════════════╣")

print(
    f"║ Predicted Final Score: {predicted_score:6.2f} / 100 ║"
)

print("║                                      ║")

print(
    f"║ Logistic Regression: {logistic_result:<15} ║"
)

print(
    f"║ KNN:                 {knn_result:<15} ║"
)

print("╚══════════════════════════════════════╝")
print("\nThank you for using ScoreSense!")
