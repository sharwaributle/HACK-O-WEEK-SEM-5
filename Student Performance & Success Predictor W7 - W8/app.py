
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression
)

from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    LabelEncoder
)

from sklearn.pipeline import make_pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ScoreSense",
    page_icon="🎓",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("🎓 ScoreSense")

st.subheader(
    "Student Performance Prediction System"
)

st.write(
    "ScoreSense uses regression and classification "
    "machine learning models to predict student "
    "performance and identify students who may need support."
)

st.divider()


# ==================================================
# LOAD DATA
# ==================================================

data = pd.read_csv(
    "data/student_data.csv"
)

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Assignments_Completed",
    "Sleep_Hours"
]


# ==================================================
# TRAIN REGRESSION MODELS
# ==================================================

X = data[features]

y = data["Final_Score"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_predictions = linear_model.predict(
    X_test
)


# Polynomial Regression

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


# Ridge

ridge_model = Ridge(
    alpha=1.0
)

ridge_model.fit(
    X_train,
    y_train
)

ridge_predictions = ridge_model.predict(
    X_test
)


# Lasso

lasso_model = Lasso(
    alpha=0.1
)

lasso_model.fit(
    X_train,
    y_train
)

lasso_predictions = lasso_model.predict(
    X_test
)


# ==================================================
# REGRESSION METRICS
# ==================================================

regression_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],

    "MAE": [
        mean_absolute_error(y_test, linear_predictions),
        mean_absolute_error(y_test, polynomial_predictions),
        mean_absolute_error(y_test, ridge_predictions),
        mean_absolute_error(y_test, lasso_predictions)
    ],

    "RMSE": [
        np.sqrt(mean_squared_error(y_test, linear_predictions)),
        np.sqrt(mean_squared_error(y_test, polynomial_predictions)),
        np.sqrt(mean_squared_error(y_test, ridge_predictions)),
        np.sqrt(mean_squared_error(y_test, lasso_predictions))
    ],

    "R2 Score": [
        r2_score(y_test, linear_predictions),
        r2_score(y_test, polynomial_predictions),
        r2_score(y_test, ridge_predictions),
        r2_score(y_test, lasso_predictions)
    ]

})


# ==================================================
# TRAIN CLASSIFICATION MODELS
# ==================================================

X_cls = data[features]

y_cls = data["Result"]


X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(

    X_cls,
    y_cls,

    test_size=0.2,

    random_state=42,

    stratify=y_cls
)


encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(
    y_train_cls
)

y_test_encoded = encoder.transform(
    y_test_cls
)


# Logistic Regression

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


# KNN

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_cls
)

X_test_scaled = scaler.transform(
    X_test_cls
)


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


# ==================================================
# CLASSIFICATION METRICS
# ==================================================

logistic_accuracy = accuracy_score(
    y_test_encoded,
    logistic_predictions
)

knn_accuracy = accuracy_score(
    y_test_encoded,
    knn_predictions
)


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


# ==================================================
# DASHBOARD — MODEL PERFORMANCE
# ==================================================

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Linear R²",
    f"{regression_results.iloc[0]['R2 Score']:.3f}"
)

col2.metric(
    "Polynomial R²",
    f"{regression_results.iloc[1]['R2 Score']:.3f}"
)

col3.metric(
    "Logistic Accuracy",
    f"{logistic_accuracy:.1%}"
)

col4.metric(
    "KNN Accuracy",
    f"{knn_accuracy:.1%}"
)


# ==================================================
# REGRESSION SECTION
# ==================================================

st.header("📈 Regression Models")

st.dataframe(
    regression_results.round(3),
    use_container_width=True
)


fig1, ax1 = plt.subplots(
    figsize=(9, 5)
)

ax1.bar(
    regression_results["Model"],
    regression_results["R2 Score"]
)

ax1.set_title(
    "Regression Model Comparison"
)

ax1.set_ylabel(
    "R² Score"
)

ax1.tick_params(
    axis="x",
    rotation=20
)

st.pyplot(fig1)


# ==================================================
# CLASSIFICATION SECTION
# ==================================================

st.header("🎯 Classification Models")

st.dataframe(
    classification_results.round(3),
    use_container_width=True
)


fig2, ax2 = plt.subplots(
    figsize=(7, 5)
)

ax2.bar(
    classification_results["Model"],
    classification_results["Accuracy"]
)

ax2.set_title(
    "Classification Model Comparison"
)

ax2.set_ylabel(
    "Accuracy"
)

ax2.set_ylim(
    0,
    1
)

st.pyplot(fig2)


# ==================================================
# CONFUSION MATRIX
# ==================================================

st.subheader(
    "Confusion Matrix — Logistic Regression"
)

cm = confusion_matrix(
    y_test_encoded,
    logistic_predictions
)


fig3, ax3 = plt.subplots(
    figsize=(5, 4)
)

ax3.imshow(cm)

ax3.set_xlabel(
    "Predicted"
)

ax3.set_ylabel(
    "Actual"
)

ax3.set_title(
    "Logistic Regression Confusion Matrix"
)

ax3.set_xticks(
    range(len(encoder.classes_))
)

ax3.set_yticks(
    range(len(encoder.classes_))
)

ax3.set_xticklabels(
    encoder.classes_
)

ax3.set_yticklabels(
    encoder.classes_
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax3.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


st.pyplot(fig3)


# ==================================================
# PREDICTION SECTION
# ==================================================

st.divider()

st.header("🔮 Predict Student Performance")

st.write(
    "Enter student information below to generate "
    "a final score prediction and classification."
)


col1, col2 = st.columns(2)


with col1:

    study_hours = st.number_input(
        "Study Hours",
        min_value=1.0,
        max_value=8.0,
        value=5.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance (%)",
        min_value=50.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    previous_score = st.number_input(
        "Previous Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0
    )


with col2:

    assignments = st.number_input(
        "Assignments Completed",
        min_value=0,
        max_value=10,
        value=7,
        step=1
    )

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=4.0,
        max_value=9.0,
        value=7.0,
        step=0.5
    )


student = pd.DataFrame({

    "Study_Hours": [study_hours],

    "Attendance": [attendance],

    "Previous_Score": [previous_score],

    "Assignments_Completed": [assignments],

    "Sleep_Hours": [sleep_hours]

})


if st.button(
    "🚀 Predict Student Performance",
    use_container_width=True
):

    # Final score

    predicted_score = linear_model.predict(
        student
    )[0]

    predicted_score = np.clip(
        predicted_score,
        0,
        100
    )


    # Logistic prediction

    logistic_prediction = logistic_model.predict(
        student
    )[0]

    logistic_result = encoder.inverse_transform(
        [logistic_prediction]
    )[0]


    # KNN prediction

    student_scaled = scaler.transform(
        student
    )

    knn_prediction = knn_model.predict(
        student_scaled
    )[0]

    knn_result = encoder.inverse_transform(
        [knn_prediction]
    )[0]


    # Display results

    st.subheader(
        "🎓 Prediction Result"
    )


    result_col1, result_col2, result_col3 = st.columns(3)


    result_col1.metric(
        "Predicted Final Score",
        f"{predicted_score:.2f} / 100"
    )


    result_col2.metric(
        "Logistic Regression",
        logistic_result
    )


    result_col3.metric(
        "KNN",
        knn_result
    )


    if (
        logistic_result == "Pass"
        and
        knn_result == "Pass"
    ):

        st.success(
            "✅ The student is predicted to PASS."
        )

    else:

        st.warning(
            "⚠️ The student may need additional academic support."
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "ScoreSense | Machine Learning Hackoweek Project"
)
