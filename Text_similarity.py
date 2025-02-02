import torch
import numpy as np
import csv
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine

# Function to calculate text similarity using transformer-based models
def calculate_similarity(model, tokenizer, text1, text2):
    # Ensure tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
    
    # Encode input texts
    encoded1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=128)
    encoded2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Compute embeddings
    with torch.no_grad():
        embedding1 = model(**encoded1).last_hidden_state.mean(dim=1).squeeze()
        embedding2 = model(**encoded2).last_hidden_state.mean(dim=1).squeeze()

    # Compute and return cosine similarity
    return 1 - cosine(embedding1.numpy(), embedding2.numpy())

# Function to apply TOPSIS for ranking models
def topsis_ranking(matrix, weights):
    # Normalize the dataset
    norm_matrix = matrix / np.linalg.norm(matrix, axis=0)

    # Apply weights
    weighted_matrix = norm_matrix * weights

    # Identify ideal and anti-ideal solutions
    ideal = np.max(weighted_matrix, axis=0)
    anti_ideal = np.min(weighted_matrix, axis=0)

    # Compute distances to ideal and anti-ideal solutions
    distance_ideal = np.linalg.norm(weighted_matrix - ideal, axis=1)
    distance_anti_ideal = np.linalg.norm(weighted_matrix - anti_ideal, axis=1)

    # Compute TOPSIS scores
    scores = distance_anti_ideal / (distance_ideal + distance_anti_ideal)

    # Rank models (higher score = better rank)
    return np.argsort(-scores) + 1  # Convert to 1-based ranking

# Define transformer models to be evaluated
models_to_test = {
    "bert-base-uncased": "BERT",
    "roberta-base": "RoBERTa",
    "sentence-transformers/bert-base-nli-mean-tokens": "SBERT",
    "gpt2": "GPT-2",
    "distilbert-base-uncased": "DistilBERT"
}

# New test sentences for similarity evaluation
sentence_1 = "The rising inflation has impacted global markets significantly."
sentence_2 = "Global markets have been heavily affected due to increasing inflation rates."

# Initialize model list and compute similarity scores
similarity_results = []
model_labels = []

for model_id, model_name in models_to_test.items():
    print(f"Processing {model_name}...")

    # Load the model and tokenizer
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Calculate similarity score
    similarity_score = calculate_similarity(model, tokenizer, sentence_1, sentence_2)

    # Store results
    similarity_results.append(similarity_score)
    model_labels.append(model_name)

    print(f"{model_name} Similarity Score: {similarity_score:.4f}")

# Convert similarity scores to a numpy array for TOPSIS processing
data_array = np.array(similarity_results).reshape(-1, 1)
weights = np.array([1])  # Single evaluation criterion (similarity)

# Apply TOPSIS to rank the models
model_rankings = topsis_ranking(data_array, weights)

# Sort results by ranking
sorted_rankings = sorted(zip(model_labels, model_rankings), key=lambda x: x[1])

# Save results to a CSV file
with open("text_similarity_model_rankings.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Rank"])
    writer.writerows(sorted_rankings)

print("\n✅ Results successfully saved to 'text_similarity_model_rankings.csv'")
