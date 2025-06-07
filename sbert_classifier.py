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

    Parameters:
        tree (dict): Category tree
        model (SentenceTransformer): SBERT model

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
    Computes a keyword-based rule score by counting keyword matches in the given description.
    Considers all keywords in the current and 1st layer of child nodes.

    Parameters:
        description (str): The input description to classify.
        node (dict): A node from the category tree, potentially with '_keywords' and subcategories.

    Returns:
        int: The cumulative score based on keyword frequency.
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
    Applies min-max normalization to an array of scores.

    Parameters:
        scores (list or np.ndarray): Raw score values to normalize.

    Returns:
        np.ndarray: Normalized scores in the range [0, 1].
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

    Parameters:
        description (str): Text description to classify.
        tree (dict): Hierarchical category tree with nested dictionaries.
        embeddings (dict): Precomputed embeddings for category labels.
        model (SentenceTransformer): SBERT model used to encode the input description.
        rule_weight (float): Weight to apply to the rule-based score relative to SBERT score.
        top_n (int): Number of top candidates to return per level.
        ambiguity_threshold (float): Threshold under which top-2 scores are considered ambiguous.

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
