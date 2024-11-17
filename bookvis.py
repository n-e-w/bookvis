import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import os
import re
from sklearn.preprocessing import StandardScaler
import warnings
from anthropic import Anthropic
from datetime import datetime
warnings.filterwarnings('ignore', category=RuntimeWarning)


def extract_json(text):
    """Extract JSON from a response that might contain additional text.
    Works for both arrays and objects."""
    try:
        # First try to parse the entire text as JSON
        try:
            return text, json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # Look for JSON array
        if '[' in text and ']' in text:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                try:
                    json.loads(json_str)
                    return json_str, json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Look for JSON object
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                try:
                    json.loads(json_str)
                    return json_str, json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        return None, None
    except Exception as e:
        print(f"Error extracting JSON: {str(e)}")
        return None, None

def normalise_text(text):
    """Normalise text by handling special characters and formatting"""
    import unicodedata
    
    # First, normalise unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    
    # Replace special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    
    # Remove language indicators and edition information
    text = re.sub(r'\s*\([^)]*(?:ingles|anglais|french|francais)[^)]*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*:\s*[^:]+$', '', text)  # Remove subtitle after colon
    text = re.sub(r'\s*-\s*Tome\s*\d+.*$', '', text, flags=re.IGNORECASE)  # Remove tome information
    
    # Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = text.strip()
    
    return text.strip()

def get_safe_title(title):
    """Get a clean version of title for JSON while preserving original for display"""
    # Remove problematic characters that might cause JSON issues
    safe = title.replace('"', "'")
    safe = re.sub(r'[^\x00-\x7F]+', '', safe)  # Remove non-ASCII characters
    safe = re.sub(r'[\n\r\t]+', ' ', safe)     # Remove newlines and tabs
    return safe.strip()

def preprocess_books(book_titles):
    """Preprocess all book titles"""
    # Create three versions of each title:
    # 1. normalised - for matching and processing
    # 2. safe - for JSON inclusion
    # 3. original - for final output
    normalised_titles = [normalise_text(title) for title in book_titles]
    safe_titles = [get_safe_title(title) for title in book_titles]
    
    # Create mappings
    title_maps = {
        'normalised_to_safe': dict(zip(normalised_titles, safe_titles)),
        'normalised_to_original': dict(zip(normalised_titles, book_titles)),
        'safe_to_original': dict(zip(safe_titles, book_titles))
    }
    
    return normalised_titles, title_maps

def get_book_titles():
    url = "https://marksaroufim.com/bookshelf"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    book_titles = []
    list_items = soup.find_all('li')
    for item in list_items:
        if not item.find_parent('nav'):
            title = item.text.strip()
            if title:
                book_titles.append(title)
    
    print(f"Found {len(book_titles)} books")
    return book_titles

def get_categorised_books(book_titles):
    # Get API key from environment variable
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("Please set ANTHROPIC_API_KEY environment variable")
    
    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)
    
    # Preprocess book titles
    normalised_titles, title_maps = preprocess_books(book_titles)
    
    # First, get the category structure
    category_prompt = """As a librarian, provide ONLY a JSON array of 8-10 category names to organise these books.
    Categories should be broad enough to handle all types of books but specific enough to be meaningful.
    
    Return ONLY the JSON array, like this:
    ["Computer Science", "Mathematics", "Fiction"]
    
    NO explanation or additional text."""
    
    try:
        # First get categories
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8192,
            messages=[{
                "role": "user",
                "content": f"{category_prompt}\n\nBook titles:\n{json.dumps(normalised_titles)}"
            }],
            temperature=0.2
        )
        
        categories_text = response.content[0].text.strip()
        json_str, categories = extract_json(categories_text)
        if not categories:
            print("Could not extract categories. Response:")
            print(categories_text)
            return None
            
        print(f"Determined categories: {categories}")
        
        # Process books in chunks
        chunk_size = 50
        all_categorised = {category: [] for category in categories}
        
        for i in range(0, len(normalised_titles), chunk_size):
            chunk = normalised_titles[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(normalised_titles) + chunk_size - 1) // chunk_size
            print(f"\nProcessing chunk {chunk_num} of {total_chunks} ({len(chunk)} books)")
            
            categorization_prompt = f"""Categorise these books using ONLY these categories:
            {json.dumps(categories)}

            Rules:
            1. EVERY book must be assigned to exactly one category
            2. Use ONLY the exact category names provided
            3. For cross-disciplinary books, use the primary subject
            4. Do not skip any books
            
            Return ONLY a JSON object mapping categories to book arrays:
            {{
                "Category1": ["Book1", "Book2"]
            }}

            NO explanations or additional text."""

            try:
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=8192,
                    messages=[{
                        "role": "user",
                        "content": f"{categorization_prompt}\n\nBooks to categorise:\n{json.dumps([title_maps['normalised_to_safe'][title] for title in chunk])}"
                    }],
                    temperature=0.1
                )
                
                chunk_text = response.content[0].text.strip()
                json_str, chunk_categories = extract_json(chunk_text)
                if not chunk_categories:
                    print(f"Could not extract JSON from chunk {chunk_num}. Response:")
                    print(chunk_text[:500] + "..." if len(chunk_text) > 500 else chunk_text)
                    continue
                
                # Validate chunk results and map back to original titles
                chunk_books = set()
                for category, books in chunk_categories.items():
                    if category not in categories:
                        print(f"Invalid category found: {category}")
                        print(f"Valid categories are: {categories}")
                        continue
                    # Map safe titles back to original titles
                    try:
                        original_titles = [title_maps['safe_to_original'][title] for title in books]
                        chunk_books.update(books)
                        all_categorised[category].extend(original_titles)
                    except KeyError as e:
                        print(f"Error mapping titles back to original: {str(e)}")
                        print(f"Problematic title: {e.args[0]}")
                        continue
                
                # Check for missing books using normalised titles
                chunk_set = set(chunk)
                safe_chunk_books = set([title_maps['normalised_to_safe'][title] for title in chunk_books])
                missing_books = chunk_set - set([title for title in chunk if title_maps['normalised_to_safe'][title] in safe_chunk_books])
                
                if missing_books:
                    print(f"Books missing from chunk {chunk_num}:")
                    for book in missing_books:
                        print(f"- {title_maps['normalised_to_original'][book]}")
                    
                    # Try to categorise missing books individually
                    print("\nAttempting to categorise missing books individually...")
                    for book in missing_books:
                        retry_prompt = f"""Choose ONE category from these options for this book:
                        {json.dumps(categories)}

Book title (choose based on subject matter): "{title_maps['normalised_to_safe'][book]}"

Return ONLY the category name, no explanation."""
                        
                        retry_response = client.messages.create(
                            model="claude-3-5-sonnet-20241022",
                            max_tokens=8192,
                            messages=[{"role": "user", "content": retry_prompt}],
                            temperature=0.1
                        )
                        
                        category = retry_response.content[0].text.strip().strip('"')
                        if category in categories:
                            all_categorised[category].append(title_maps['normalised_to_original'][book])
                            print(f"Categorised '{title_maps['normalised_to_original'][book]}' as '{category}'")
                        else:
                            print(f"Failed to categorise '{title_maps['normalised_to_original'][book]}'")
                            continue
                
                print(f"Successfully processed chunk {chunk_num}")
                
            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {str(e)}")
                print("Response received:")
                print(response.content[0].text.strip())
                continue
        
        # Final validation using original titles
        all_categorised_books = set()
        for category, books in all_categorised.items():
            all_categorised_books.update(books)
        
        missing_books = set(book_titles) - all_categorised_books
        if missing_books:
            print("\nBooks missing from final categorisation:")
            for book in list(missing_books)[:10]:
                print(f"- {book}")
            if len(missing_books) > 10:
                print(f"...and {len(missing_books) - 10} more")
        
        print(f"\nFinal Statistics:")
        print(f"Total categories: {len(all_categorised)}")
        print(f"Total books categorised: {len(all_categorised_books)} out of {len(book_titles)}")
        print("\nBooks per category:")
        for category, books in sorted(all_categorised.items()):
            print(f"{category}: {len(books)} books")
            
        return all_categorised
            
    except Exception as e:
        print(f"Error in categorisation: {str(e)}")
        if 'response' in locals():
            print("\nLast response received:")
            print(response.content[0].text.strip())
        return None

    
def preprocess_title(title):
    title = re.sub(r': .*Edition.*$', '', title)
    title = re.sub(r'Vol\. \d+', '', title)
    return title

def extract_books(d):
    books = []
    categories = []
    subcategories = []
    
    for category, items in d.items():
        if isinstance(items, list):
            books.extend(items)
            categories.extend([category] * len(items))
            subcategories.extend([category] * len(items))
    
    print(f"Extracted {len(books)} books across {len(set(categories))} categories")
    for cat in set(categories):
        count = categories.count(cat)
        print(f"  {cat}: {count} books")
    
    return books, categories, subcategories

def create_visualisation(books, categories, subcategories, embeddings):
    # Set random seed for reproducibility
    np.random.seed(110)
    
    # Apply UMAP with improved parameters and random seed
    print("Applying UMAP...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=20,
        min_dist=0.4,
        metric='cosine',
        random_state=110,  # Added fixed random state
        spread=2.0,
        n_epochs=750,
        repulsion_strength=2.0
    )
    coords = reducer.fit_transform(embeddings)

    # Define a color palette
    color_palette = px.colors.qualitative.Set3[:8]  # Using plotly's Set3 palette
    category_colors = {cat: color_palette[i % len(color_palette)] 
                      for i, cat in enumerate(sorted(set(categories)))}

    # Create traces for each category
    traces = []
    for category in sorted(set(categories)):
        indices = [i for i, cat in enumerate(categories) if cat == category]
        
        trace = go.Scatter3d(
            x=coords[indices, 0],
            y=coords[indices, 1],
            z=coords[indices, 2],
            mode='markers',
            name=category,
            hovertext=[f"Title: {books[i]}<br>Category: {category}" 
                      for i in indices],
            hoverinfo='text',
            marker=dict(
                size=8,
                opacity=0.65,
                color=category_colors[category],
                line=dict(width=1, color='white'),
                symbol='circle'
            )
        )
        traces.append(trace)

    # Create figure with improved layout
    fig = go.Figure(data=traces)

    # Update layout with improved parameters
    fig.update_layout(
        title={
            'text': 'Book Topics 3D Visualisation (UMAP)',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=2.0, y=2.0, z=1.8)
            ),
            aspectmode='data'
        ),
        width=2400,
        height=1600,
        showlegend=True,
        legend=dict(
            title=dict(
                text='Book Categories',
                font=dict(size=16)
            ),
            itemsizing='constant',
            itemwidth=30,
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1
        ),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig

def main():
    # Get book titles
    print("Fetching book titles...")
    book_titles = get_book_titles()

    # Get categorised structure from API
    print("Categorising books using Claude...")
    clusters = get_categorised_books(book_titles)
    
    if not clusters:
        print("Failed to get categorisation from Claude API")
        return

    # Extract and preprocess books
    print("Extracting books...")
    book_titles, book_categories, book_subcategories = extract_books(clusters)
    processed_titles = [preprocess_title(title) for title in book_titles]

    # Generate embeddings
    print("Generating embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(processed_titles, show_progress_bar=True)

    # Normalize embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Create visualization
    fig = create_visualisation(book_titles, book_categories, book_subcategories, normalized_embeddings)

    # Show the figure
    fig.show()

    # Save to HTML with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.abspath(f"book_visualisation_umap_{timestamp}.html")
    fig.write_html(output_path)
    print(f"Visualisation saved to: {output_path}")

if __name__ == "__main__":
    np.random.seed(110)
    main()