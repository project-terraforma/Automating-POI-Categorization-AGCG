#(C) Adam Axtopani Gonzales - SPR 2025

"""
This file contains the core functions for embedding and classifying category descriptions
using a Sentence-BERT model within a hierarchical taxonomy structure. It supports:
- Layer-wise node embedding
- Combined embedding + rule-based classification
- Ambiguity detection for closely scoring candidates

Used as part of a multi-layer category prediction system.
"""
from category_keywords import category_keywords
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import numpy as np
import re


def embed_tree_nodes_by_layer(tree, model):
    """
    Traverses the hierarchical tree and encodes every node label using SBERT.
    Avoids encoding "_keywords" and skips duplicate keys.

    Args:
        tree (dict): The hierarchical category tree
        model (SentenceTransformer): Pre-loaded SBERT model

    Returns:
        dict: Dictionary mapping node labels to embedding vectors
    """
    embeddings = {}

    def recurse(node):
        for key, value in node.items():
            if key == "_keywords":
                continue  # Skip keyword metadata

            # Only embed unique keys
            if key not in embeddings:
                try:
                    embeddings[key] = model.encode(key)
                except Exception as e:
                    print(f"Error encoding key '{key}': {e}")
                    continue

            if isinstance(value, dict):
                recurse(value)

    recurse(tree)
    return embeddings


def get_rule_score(description, node):
    """
    Computes a keyword matching score between the input description and the keywords
    found in the current and all descendant nodes of the tree.

    Returns:
        int: The total number of keyword matches found (can count duplicates)
    """
    desc = description.lower()
    score = 0

    def collect_keywords(n):
        if not isinstance(n, dict):
            return []
        collected = list(n.get("_keywords", []))
        for child in n:
            if child != "_keywords" and isinstance(n[child], dict):
                collected += collect_keywords(n[child])
        return collected

    all_keywords = collect_keywords(node)
    word_freq = Counter(re.findall(r'\b\w+\b', desc))

    for keyword in all_keywords:
        if ' ' in keyword:
            score += desc.count(keyword.lower())
        else:
            score += word_freq[keyword.lower()]

    return score


def normalize_scores(scores):
    """
    Min-max normalization of scores to [0, 1] range.

    Returns:
        np.array: Normalized scores
    """
    min_s = np.min(scores)
    max_s = np.max(scores)
    if max_s == min_s:
        return np.ones_like(scores)
    return (scores - min_s) / (max_s - min_s)


def classify_with_layered_tree_top_n(
    description,
    tree,
    embeddings,
    model,
    rule_weight=0.5,
    top_n=3,
    ambiguity_threshold=0.1
):
    """
    Classifies a description down the tree by evaluating each layer's children
    using both semantic (SBERT) and rule-based scores.

    Returns:
        str: Final predicted path as string
        list: Top-N candidates at each level
        bool: Whether any layer was ambiguous
        list: Layer indices where ambiguity occurred
    """
    desc_embedding = model.encode(description)
    current_node = tree
    current_path = []
    full_result = []
    ambiguous_layers = []

    while True:
        children = [k for k in current_node if k != "_keywords"]
        if not children:
            break

        child_vectors = [embeddings[c] for c in children]
        sims = util.cos_sim(desc_embedding, child_vectors)[0].cpu().numpy()
        rule_scores = [get_rule_score(description, current_node[c]) for c in children]

        combined_scores = sims + rule_weight * np.array(rule_scores)
        combined_scores = normalize_scores(combined_scores)

        # Ambiguity detection: if top 2 are too close
        sorted_scores = np.sort(combined_scores)[::-1]
        if len(sorted_scores) > 1 and abs(sorted_scores[0] - sorted_scores[1]) < ambiguity_threshold:
            ambiguous_layers.append(len(full_result))

        # Get top N children for display
        ranked_indices = np.argsort(combined_scores)[::-1][:top_n]
        ranked_children = [(children[i], combined_scores[i]) for i in ranked_indices]
        full_result.append(ranked_children)

        best_child = ranked_children[0][0]
        current_path.append(best_child)
        current_node = current_node[best_child]

    is_ambiguous = len(ambiguous_layers) > 0
    return " > ".join(current_path), full_result, is_ambiguous, ambiguous_layers
