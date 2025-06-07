#(C) Adam Axtopani Gonzales - SPR 2025

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin  # Ensures clean joining of base URLs with relative paths

def find_about_page(base_url):
    """
    Attempts to locate the 'About' page for a given website.

    The function first tries a set of common URL paths (e.g., '/about', '/company').
    If none of those return a valid response, it falls back to parsing the site's homepage
    to find anchor tags that may point to an About page.

    Parameters:
        base_url (str): The base URL of the website (e.g., "https://example.com").

    Returns:
        str or None: The full URL to the About page if found, otherwise None.
    """
    
    # List of common About page paths to check first
    common_paths = [
        '/about', '/about-us', '/aboutus',
        '/company', '/who-we-are', '/our-story',
        '/story'
    ]
    
    # Step 1: Try known About page paths directly
    for path in common_paths:
        full_url = urljoin(base_url, path)  # Join base URL with path cleanly
        try:
            response = requests.get(full_url, timeout=5)
            if response.status_code == 200:
                return full_url  # Return the first matching page that exists
        except requests.RequestException:
            # Ignore failed attempts (e.g., 404, timeout) and try the next one
            continue

    # Step 2: Fallback - parse homepage for links that contain 'about' or 'company'
    try:
        homepage = requests.get(base_url, timeout=5)
        soup = BeautifulSoup(homepage.text, 'html.parser')

        # Look through all anchor tags
        for a in soup.find_all('a', href=True):
            href = a['href'].lower()
            text = a.text.lower()

            # If the href or link text contains indicators of an About page
            if 'about' in href or 'about' in text or 'company' in href or 'company' in text:
                return urljoin(base_url, a['href'])  # Return fully joined URL
    except requests.RequestException:
        # Catch any issues with requesting the homepage
        pass

    # If nothing matches, return None
    return None

def extract_main_text(url, min_len=300):
    """
    Extracts the main block of descriptive text content from a webpage.

    This function performs a quick heuristic-based extraction by searching
    for the first sufficiently large and unique block of text within common
    content tags (e.g., <p>, <div>, <section>, <article>). Boilerplate elements
    like headers, footers, scripts, and forms are removed before parsing.

    Parameters:
        url (str): The full URL of the webpage to extract content from.
        min_len (int): Minimum character length required for a text block to be considered (default is 300).

    Returns:
        str: The extracted text block if found, or an error/sentinel string.
    """

    try:
        # Fetch the HTML content from the URL
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Remove non-content or boilerplate elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'form']):
            tag.decompose()

        # Keep track of already-seen text blocks to avoid duplication
        seen = set()

        # Search through common content tags
        for tag in soup.find_all(['p', 'div', 'section', 'article']):
            text = tag.get_text(strip=True)

            # Check if text is long enough and hasn't been seen before
            if len(text) >= min_len and text not in seen:
                seen.add(text)
                return text  # Return the first valid block

        return "[No suitable block found]"

    except Exception as e:
        # Handle any error that may occur during request or parsing
        return f"[ERROR] {e}"

def extract_meta_and_title(url):
    """
    Extracts the page title and meta description from a webpage.

    This function fetches the HTML content of a given URL and parses it to retrieve:
        - The <title> tag content (usually shown in browser tabs or search results)
        - The <meta name="description"> tag content (commonly used as a short summary in search engines)

    Parameters:
        url (str): The URL of the webpage to extract metadata from.

    Returns:
        dict: A dictionary containing:
              - 'title': The content of the <title> tag (str)
              - 'meta_description': The content of the <meta name="description"> tag (str),
                                     or an error message if something fails.
    """
    try:
        # Send an HTTP GET request to fetch the page content
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')

        # Extract and clean <title> tag text, if present
        title = soup.title.string.strip() if soup.title else ""

        # Attempt to find the <meta name="description"> tag
        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'})

        # If the meta tag is found and contains a content attribute, extract and clean it
        if meta_tag and meta_tag.get('content'):
            meta_desc = meta_tag['content'].strip()

        return {
            "title": title,
            "meta_description": meta_desc
        }

    except Exception as e:
        # On failure, return error in the meta_description field for debugging
        return {
            "title": "",
            "meta_description": f"[ERROR] {e}"
        }

