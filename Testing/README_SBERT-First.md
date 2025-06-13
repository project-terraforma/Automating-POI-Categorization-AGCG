Overture POI Categorization Pipeline

This project explores methods for automatically categorizing Points of Interest (POIs) from the Overture Maps Foundation dataset. The goal is to develop a robust pipeline that can assign accurate categories to POIs, especially those with missing or ambiguous information.

Project Strategy:
The primary strategy is a multi-stage classification pipeline that combines a modern semantic search approach with a traditional rule-based system. This hybrid model aims to maximize both coverage and accuracy.

The pipeline operates as follows:

1) SBERT-First Approach: For each POI, the system first attempts to classify it using a Sentence Transformer model (all-MiniLM-L6-v2).
2) Enhanced Text Retrieval: To provide rich text to the SBERT model, the pipeline uses an enhanced scraping system:
        - It first tries to scrape the official website URL provided in the Overture data using the trafilatura library and a fallback to the site's meta description.
        - If the primary URL fails or yields insufficient text, a fallback mechanism performs a web search using duckduckgo-search to find a relevant website, which is then scraped.
3) Rule-Based Fallback: If the SBERT model cannot make a confident prediction (e.g., no text could be found or the semantic similarity score is too low), the system falls back to a keyword-matching classifier based on the Overture category taxonomy.
4) Evaluation: The final combined predictions are evaluated against Overture's primary_category to measure accuracy and the effectiveness of each pipeline stage.
How to Run This Project


1. Prerequisites
You will need a Python environment with the required libraries installed. It is highly recommended to use a virtual environment.
    - Python 3.9+
    - PIP (Python package installer)
2. Setup
Clone the Repository:

git clone <your-repository-url>
cd <your-repository-directory>
Create and Activate a Virtual Environment (Recommended):


# Create the environment
python3 -m venv maps_env

# Activate the environment
source maps_env/bin/activate


Install Required Libraries:
Run the following command in your activated virtual environment to install all necessary packages.


pip install pandas numpy overturemaps sentence-transformers torch scikit-learn trafilatura duckduckgo-search beautifulsoup4 requests jupyterlab


3. Running the Notebook
4. 
The main logic and experimentation are contained in the Jupyter Notebooks.

Launch Jupyter Lab: From your terminal (with the virtual environment activated), run:
Bash

jupyter lab

Open the Notebook: In the Jupyter Lab interface that opens in your browser, open the SBERT-First.ipynb notebook file.

Ensure Data Files are Present: Make sure the Overture category taxonomy file is in the same directory as the notebook. The notebook currently expects it to be named:
overture_categories.csv ( check cell [5] in the notebook and ensure the filename matches).


Execute Cells in Order: To ensure the pipeline runs correctly, run the notebook cells from top to bottom. You can do this by selecting "Run" -> "Run All Cells" from the menu, or by running each cell individually in sequence.

Project Structure
- SBERT-First.ipynb: The main notebook containing the SBERT-first classification pipeline, evaluation, and documentation.
- overture_categories.csv: The Overture category taxonomy file used to generate category embeddings and rule-based keywords.
- scraping.ipynb: (Optional) An earlier notebook containing development work on the web scraping functions.


Current OKRs (Objectives and Key Results)

Objective 1: Establish a working baseline pipeline for POI categorization. (Completed)
KR 1.1: Load and explore a representative POI dataset for San Francisco. (completed)
KR 1.2: Apply basic name-cleaning and heuristic rules to categorize POIs. (completed)
KR 1.3: Identify and document rules in a testable Python dictionary and function. (completed)

Objective 2: Integrate ML or LLM Techniques to Improve POI Categorization. (In Progress
KR 2.1: Select and evaluate at least two distinct ML or LLM-based classification approaches. (The SBERT-first approach is the primary method being evaluated in SBERT-First.ipynb). (completed)
KR 2.2: Construct and manually label a high-quality evaluation dataset of at least 200 POIs. (completed)
KR 2.3: Achieve a classification accuracy of at least 40-60% using a chosen ML/LLM model on the new evaluation dataset.