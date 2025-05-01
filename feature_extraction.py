import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

df = df.dropna(subset=['cleaned_text'])  # Drop NaNs before processing
df.to_csv("cleaned_data.csv", index=False)  # Save cleaned data properly

# Use TF-IDF to convert text into numbers
tfidf = TfidfVectorizer(max_features=5000)  # Keep top 5000 words
X = tfidf.fit_transform(df['cleaned_text'])  # Convert text to TF-IDF features

# Convert target labels (0 = negative, 4 = positive)
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)  # Convert 4 to 1 (positive) and 0 remains (negative)
y = df['target']

# Save TF-IDF model (optional)
import pickle
pickle.dump(tfidf, open("tfidf_model.pkl", "wb"))  # Save the model for later use

print("TF-IDF transformation complete!")
print("Shape of TF-IDF matrix:", X.shape)
