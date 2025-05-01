import pandas as pd

# Define the model results (replace with actual results from training)
results = {
    "Model": ["Logistic Regression", "SVM", "LSTM"],
    "Accuracy": [0.7744, 0.7613, 0.7622],
    "Precision": [0.79, 0.76, 0.76],
    "Recall": [0.75, 0.72, 0.80],
    "F1-Score": [0.77, 0.76, 0.77],
}

# Convert to DataFrame
df = pd.DataFrame(results)

# Save the report as a CSV file
df.to_csv("model_comparison_report.csv", index=False)

# Print the report
print("Report generated successfully check it in the folder.")
print(df)
