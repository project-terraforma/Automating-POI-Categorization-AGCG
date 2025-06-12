# Automating POI Categorization

A hybrid system combining semantic embeddings and rule-based logic to classify Points of Interest (POIs) using a hierarchical category tree.

## Description

This project is designed to automatically categorize POIs (like restaurants, gyms, or clinics) into a structured taxonomy. POI data is collected by scraping publicly available information from business websites. For each POI, we use the business name and website content (extracted via web scraping) to generate a descriptive input.

The model then uses Sentence-BERT (SBERT) embeddings to match this input against a tree of categories and subcategories. This is enhanced by a rule-based scoring system that matches category-specific keywords to improve prediction accuracy. This combination ensures scalable, interpretable, and flexible POI classification, especially when dealing with sparse or noisy data.

## Getting Started

### Dependencies

- **Operating System**: Windows 10, macOS, or Linux (Python 3.8+ recommended)
- **Python Libraries**:
  - `transformers` – for loading sentence embedding models
  - `sentence-transformers` – high-level wrapper for semantic embedding
  - `torch` – backend for running SBERT model inference
  - `pandas` – used for handling and filtering POI datasets
  - `numpy` – array computations for scoring and embedding math
  - `beautifulsoup4` – used for HTML parsing in the web scraper
  - `requests` – makes HTTP calls to fetch POI websites
  - `jupyter` – local interactive development
  - `google-colab` – cloud-based alternative to Jupyter Notebooks
 
### Setup
1. Clone the repo
   ```sh
   git clone https://github.com/project-terraforma/Automating-POI-Categorization-AGCG.git
   ```
2. Navigate into the project directory
   ```sh
   cd Automating-POI-Categorization-AGCG
   ```
3. Install Python dependencies

   Make sure you have a Python environment set up, then install required packages:
   ```sh
   pip install -r requirements.txt
   ```
5. Prevent accidental pushes to the base repository

   Change the Git remote to your own fork or local version:
   ```sh
   git remote set-url origin https://github.com/<your-username>/<your-repo-name>.git
   git remote -v  # Confirm the remote URL was updated
   ```
8. Start the project

   Launch Jupyter Notebook and open the main notebook to begin:
   ```sh
   jupyter notebook
   ```

   Navigate to the notebooks/ folder and open main.ipynb.

## Authors

Adam Axtopani Gonzales – adamurlnum2@gmail.com

Carlos Garcia

## Version History

* 0.1
    * Initial Release


## Acknowledgments

### [ Project Sponsor ] Overture Maps Foundation
Sposnored this project and gave us the opportunity to approach this problem with their open source data

### Overture Maps POC's from Microsoft Corporation 

- Krill Fedotov, Marko Radoicic, & Nikola Bozovic
A source of guidance and expertise when tackling this project together


