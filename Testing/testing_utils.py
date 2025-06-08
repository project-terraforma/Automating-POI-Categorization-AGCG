import pandas as pd
import overturemaps
import web_scraper as webScraper


def fetch_overture_poi_data(theme: str, bbox: tuple) -> pd.DataFrame:
    """
    Fetches POI data from Overture Maps for a given theme and bounding box.

    Parameters:
        theme (str): The theme to fetch (e.g., 'place').
        bbox (tuple): Bounding box as (min_lon, min_lat, max_lon, max_lat).

    Returns:
        pd.DataFrame: DataFrame containing the POI data.
    """
    print(f"\nBounding box: {bbox}")
    print(f"Theme: '{theme}'")
    print("Fetching data from Overture Maps...")

    try:
        reader = overturemaps.record_batch_reader(theme, bbox=bbox)
        df = reader.read_all().to_pandas()

        print(f"Successfully loaded {len(df)} POIs.")
        return df

    except Exception as e:
        print(f"Error: {e}")
        print("Check your internet connection or Overture Maps availability.")
        return pd.DataFrame()
    
def scrape_website_batch(df_websites, webScraper, max_sites=100, min_meta_len=75, min_about_len=100, progress_every=10):
    """
    Scrapes a batch of websites from a DataFrame using meta and about-page heuristics.

    Parameters:
        df_websites (DataFrame): Must contain 'names', 'categories', and 'websites' columns.
        webScraper (module): Custom web scraper module containing `extract_meta_and_title`, `find_about_page`, and `extract_main_text`.
        max_sites (int): Number of websites to process from the DataFrame.
        min_meta_len (int): Minimum length of meta description to be considered valid.
        min_about_len (int): Minimum length of about/main text to be considered valid.
        progress_every (int): Frequency of printing progress (e.g., every 10 websites).

    Returns:
        List[dict]: Each dictionary contains site metadata and extracted text content.
    """
    results = []

    for i, raw in enumerate(df_websites['websites']):
        if i >= max_sites:
            break

        base_url = raw[0]
        name = df_websites.iloc[i]['names']
        category = df_websites.iloc[i]['categories']

        record = {
            "name": name,
            "category": category,
            "url": base_url,
            "text": "",
            "status": "incomplete",
            "source": []
        }

        try:
            # Try meta extraction
            meta_info = webScraper.extract_meta_and_title(base_url)
            title = meta_info.get("title", "").strip()
            meta_desc = meta_info.get("meta_description", "").strip()

            if not meta_desc.startswith("[ERROR]") and len(meta_desc) >= min_meta_len:
                record["text"] += title + "\n" + meta_desc + "\n"
                record["source"].append("meta")
        except Exception:
            record["source"].append("meta_error")

        try:
            # Try to find and scrape the about page
            about_url = webScraper.find_about_page(base_url) or base_url
            about_text = webScraper.extract_main_text(about_url)

            if not about_text.startswith("[ERROR]") and len(about_text) >= min_about_len:
                record["text"] += about_text.strip()
                record["source"].append("about" if about_url != base_url else "fallback")
        except Exception:
            record["source"].append("about_error")

        # Final status determination
        if record["text"].strip():
            record["status"] = "success"
        else:
            record["status"] = "no_valid_text"

        results.append(record)

        # Progress report
        if (i + 1) % progress_every == 0:
            print(f"Processed {i + 1} websites")

    print("\nScraping completed.")
    return results

def extract_row_info(df, index=0):
    """
    Extracts name, categories (primary + alternate), text content, and status from a given row.

    Parameters:
        df (pd.DataFrame): The DataFrame containing scraped website data.
        index (int): Index of the row to extract (default is 0).

    Returns:
        dict: Dictionary with:
            - name (str): Primary name
            - categories (list): Combined list of primary and alternate categories
            - text (str): Combined meta/about page description
            - status (str): Scrape status
    """
    try:
        row = df.iloc[index]

        name = row["name"].get("primary", "N/A")
        primary = row["category"].get("primary", "")
        alternate = row["category"].get("alternate", [])

        categories = [primary] + list(alternate)

        return {
            "name": name,
            "categories": categories,
            "text": row.get("text", ""),
            "status": row.get("status", "")
        }

    except Exception as e:
        print(f"[ERROR] Could not extract row info: {e}")
        return None
