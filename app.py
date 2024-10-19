from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

app = Flask(__name__)

# Load documents (replace with your actual digital marketing documents)
documents = [
    "Digital marketing strategies in 2024.pdf",
    "SEO optimization techniques.pdf",
    "Content marketing and engagement.pdf",
    "The role of social media in digital marketing.pdf",
    # Add more document names here
]

# Placeholder for document content (cleaned text, not raw PDFs)
document_content = [
    "Text content of Digital marketing strategies in 2024.",
    "Text content of SEO optimization techniques.",
    "Text content of Content marketing and engagement.",
    "Text content of The role of social media in digital marketing.",
    # Add text content for all documents here
]

# Dummy labels for each document (1 = relevant, 0 = irrelevant, replace as needed)
document_labels = [1, 1, 0, 0]  # Add your own relevance labels here

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(document_content)

# Function to calculate Precision at K
def precision_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    precision = len(relevant_retrieved) / k
    return precision

# Function to calculate Recall at K
def recall_at_k(retrieved_docs, relevant_docs, k):
    retrieved_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_k if doc in relevant_docs]
    recall = len(relevant_retrieved) / len(relevant_docs) if relevant_docs else 0
    return recall

# Logistic Regression evaluation function
def evaluate_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)  # Train the model
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')

    return accuracy, precision, recall

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between the query and documents
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()

    # Get top 5 most similar documents
    top_indices = similarity_scores.argsort()[-5:][::-1]
    retrieved_docs = [documents[i] for i in top_indices]

    # Placeholder for relevant documents (use actual relevant docs logic here)
    relevant_docs = [documents[0], documents[1]]  # Example of relevant docs

    # Calculate Precision and Recall at K (k=5)
    precision_at_k_value = precision_at_k(retrieved_docs, relevant_docs, k=5)
    recall_at_k_value = recall_at_k(retrieved_docs, relevant_docs, k=5)

    # Evaluate using Logistic Regression
    accuracy, precision, recall = evaluate_model(doc_vectors.toarray(), document_labels)

    return render_template('result.html', 
                           query=query, 
                           results=[(documents[i], similarity_scores[i]) for i in top_indices],
                           precision_at_k=precision_at_k_value, 
                           recall_at_k=recall_at_k_value,
                           accuracy=accuracy, 
                           precision=precision, 
                           recall=recall)


if __name__ == '__main__':
    app.run(debug=True)
