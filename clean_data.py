import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords  

# Load dataset (Make sure "training.1600000.processed.noemoticon.csv" is in the same folder)
df = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)

# Assign column names
df.columns = ["target", "ids", "date", "flag", "user", "text"]

# Download stopwords (only needed once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions (@username)
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text  

# Apply cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Show first 5 rows
print(df[['text', 'cleaned_text']].head())

# Save cleaned data to a new CSV file
df.to_csv("cleaned_data.csv", index=False)
