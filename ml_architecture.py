import matplotlib.pyplot as plt
import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Define edges
edges = [
    ("Input (Text)", "TF-IDF Vectorizer"),
    ("TF-IDF Vectorizer", "Logistic Regression"),
    ("TF-IDF Vectorizer", "SVM"),
    ("Logistic Regression", "Sentiment Output"),
    ("SVM", "Sentiment Output"),
]

# Add edges to graph
G.add_edges_from(edges)

# Plot the graph
plt.figure(figsize=(8, 5))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=10)

plt.title("Machine Learning Model Architecture")
plt.savefig("ml_architecture.png")  # Save as PNG
plt.show()

print("Machine Learning Architecture saved as ml_architecture.png")
