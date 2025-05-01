import pandas as pd  

# Load dataset (Change file path if needed)
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)

# Assign column names
df.columns = ["target", "ids", "date", "flag", "user", "text"]

# Show first 5 rows
print(df.head())
