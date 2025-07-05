#!/usr/bin/env python3
import os
from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search to find the most similar document in the corpus to the input sentence.

    Args:
        corpus_path (str): path to directory of reference documents (.md, .txt, etc.)
        sentence (str): the query sentence to match against corpus

    Returns:
        str: content of the most similar document
    """
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load and encode all corpus documents
    documents = []
    for filename in os.listdir(corpus_path):
        path = os.path.join(corpus_path, filename)
        if os.path.isfile(path) and filename.endswith(('.md', '.txt')):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(content)

    # Encode corpus and input sentence
    doc_embeddings = model.encode(documents, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

    # Get best match index
    best_match_idx = similarities.argmax().item()
    return documents[best_match_idx]
