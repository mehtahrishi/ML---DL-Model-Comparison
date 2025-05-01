import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load TF-IDF model
tfidf = pickle.load(open("tfidf_model.pkl", "rb"))

# Load cleaned dataset
df = pd.read_csv("cleaned_data.csv")

# Use only a subset (reduce training time)
df = df.sample(n=200000, random_state=42)  # Using 200,000 samples instead of 1.5M

# Transform text using TF-IDF
X = tfidf.transform(df['cleaned_text'])  
y = df['target'].apply(lambda x: 1 if x == 4 else 0)  # Convert target labels

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an optimized SVM using SGDClassifier (MUCH FASTER!)
model = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# Save trained model
pickle.dump(model, open("svm_model.pkl", "wb"))

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("SVM training complete!")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification report:\n", classification_report(y_test, y_pred))
