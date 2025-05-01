from graphviz import Digraph

# Initialize Graph
dot = Digraph()

# Define Nodes
dot.node('A', 'Start')
dot.node('B', 'Load Dataset')
dot.node('C', 'Preprocessing (Lowercase, Remove Punctuation)')
dot.node('D', 'Tokenization & Stopword Removal')
dot.node('E', 'TF-IDF Vectorization')
dot.node('F', 'Train/Test Split')
dot.node('G', 'End')

# Define Edges
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])

# Save and render diagram
dot.render('flowchart', format='png', cleanup=False)
print(" Flowchart saved as flowchart.png")
