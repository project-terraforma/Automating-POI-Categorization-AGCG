#(C) Adam Axtopani Gonzales - SPR 2025

import json

# Load the category tree from file
with open("data/category_tree.json", "r") as f:
    category_tree = json.load(f)

"""
Prediction Testing Utilities
These helper functions are used to validate and analyze predictions made by the 
SBERT-based classification system. They help determine:
    - The full path of a predicted category
    - Whether the top-level predicted category matches the true label
    - Useful for evaluation pipelines and error analysis
"""

def find_category_path(tree, target_category, path=None):
    """
    Recursively find the full path to a specific category in the tree.

    Parameters:
        tree (dict): Category tree.
        target_category (str): Target leaf or node.
        path (list): Current recursive path (used internally).

    Returns:
        list or None: Full path as a list of category names, or None if not found.
    """

    if path is None:
        path = []

    for category, children in tree.items():
        new_path = path + [category]
        if category == target_category:
            return new_path
        if isinstance(children, dict):
            result = find_category_path(children, target_category, new_path)
            if result:
                return result
    return None

def find_top_level_category(tree, target_category, path=None):
    """
    Find the top-level category that a target subcategory belongs to.

    Parameters:
        tree (dict): Category tree.
        target_category (str): Leaf or node.

    Returns:
        str or None: Top-level category name.
    """

    if path is None:
        path = []

    for category, children in tree.items():
        new_path = path + [category]
        if category == target_category:
            return path[0] if path else category
        if isinstance(children, dict):
            result = find_top_level_category(children, target_category, new_path)
            if result:
                return result
    return None

def is_prediction_correct(predicted_top_category, true_category_label):
    """
    Check whether the predicted top-level category matches the true category label.
    This is used for model testing/evaluation.

    Parameters:
        predicted_top_category (str): Model output.
        true_category_label (str): True label (can be subcategory).

    Returns:
        bool: True if the prediction is correct at the top level.
    """

    true_top_category = find_top_level_category(category_tree, true_category_label)
    return predicted_top_category == true_top_category

"""
Tree Exploration Utilities:
  These functions are designed to help developers and contributors explore and navigate
  the hierarchical category tree. They assist in:
    - Browsing top-level and nested categories
    - Counting subcategories at each level
    - Extracting subtrees to focus on keyword creation
"""

def get_subcategories(tree, path):
    """
    Traverse the tree and return the sub-tree for a given category path.
    
    Parameters:
        tree (dict): The category tree.
        path (list): A list of nested keys representing the path.

    Returns:
        dict: Sub-categories at the specified "node" category.
    """

    current = tree
    for key in path:
        if key in current:
            current = current[key]
        else:
            return {}
    return current  # returns sub-tree

def test_top_level_categories(tree):
    """
    Print all top-level categories for overview/debugging.

    Parameters:
        tree (dict): The category tree.
    """
    for top_level in tree.keys():
        print(f"- {top_level}")

def count_and_sort_subcategories(category_keywords):
    """
    Count and sort categories by number of immediate subcategories.

    Parameters:
        tree (dict): The full category tree.

    Returns:
        dict: Sorted dictionary with counts.
    """

    subcategory_counts = {}
    
    for top_level, data in category_keywords.items():
        count = sum(1 for key in data if key != "_keywords")
        subcategory_counts[top_level] = count

    # Sort by subcategory count (ascending)
    sorted_counts = dict(sorted(subcategory_counts.items(), key=lambda item: item[1]))
    return sorted_counts