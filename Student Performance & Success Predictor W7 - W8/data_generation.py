import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)


# Make results reproducible
np.random.seed(42)

# Number of students
n_students = 1000

# Generate student information
study_hours = np.random.uniform(1, 8, n_students)
attendance = np.random.uniform(50, 100, n_students)
previous_score = np.random.uniform(35, 95, n_students)
assignments = np.random.randint(2, 11, n_students)
sleep_hours = np.random.uniform(4, 9, n_students)

# Calculate final score
final_score = (
    0.35 * previous_score
    + 2.5 * study_hours
    + 0.25 * attendance
    + 1.5 * assignments
    + 1.2 * sleep_hours
    + np.random.normal(0, 5, n_students)
)

# Keep scores between 0 and 100
final_score = np.clip(final_score, 0, 100)

# Create Pass / Needs Support label
result = np.where(
    final_score >= 40,
    "Pass",
    "Needs Support"
)

# Create dataframe
data = pd.DataFrame({
    "Study_Hours": study_hours.round(2),
    "Attendance": attendance.round(2),
    "Previous_Score": previous_score.round(2),
    "Assignments_Completed": assignments,
    "Sleep_Hours": sleep_hours.round(2),
    "Final_Score": final_score.round(2),
    "Result": result
})

# Save dataset
data.to_csv(
    "data/student_data.csv",
    index=False
)

print("Dataset created successfully!")
print()
print(data.head())
print()
print("Dataset shape:", data.shape)
