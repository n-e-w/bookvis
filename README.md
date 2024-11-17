# Book Visualisation

Based on the following [discussion thread on X.](https://x.com/renegadesilicon/status/1857767141375881257) Thanks to [Alexa Gordic](https://x.com/gordic_aleksa) and [Mark Saroufim](https://x.com/marksaroufim) for the inspitation for putting this together.

Creates an interactive 3D visualisation of book relationships using UMAP dimensionality reduction and sentence embeddings. By default, it visualises books from Mark Saroufim's bookshelf.

Note: this is a bare bones implementation. It only works with Mark's website and it's all hardcoded (though trivial to fix this if there's interest). I've not packaged it up as a library or anything. It's pretty straightforward to read and interpret. If there's interest in scaling / extending this I'll fork it to a proper `nbdev` project. Please feel free to send PRs.

## How It Works

1. **Data Collection**: 
   - Scrapes book titles from marksaroufim.com/bookshelf
   - Preprocesses titles to remove editions, volumes, and normalise text

2. **Categorisation**:
   - Uses Claude AI to create 8-10 meaningful categories
   - Processes books in chunks of 50 for efficient API usage
   - Handles edge cases and missing categorisations

3. **Embedding Generation**:
   - Uses the `all-mpnet-base-v2` model from SentenceTransformers
   - Chosen for its strong performance on semantic similarity tasks
   - Generates 768-dimensional embeddings for each book title
   - Model parameters:
     - `max_seq_length=128`: Maximum sequence length for input text
     - `normalize_embeddings=True`: L2 normalises embeddings for better similarity comparison
     - `batch_size=32`: Processes multiple titles efficiently

4. **Dimensionality Reduction**:
   - Uses UMAP (Uniform Manifold Approximation and Projection)
   - Parameters optimised for book visualisation:
     - `n_components=3`: Output dimensionality for 3D visualisation
     - `n_neighbors=20`: Balances local and global structure
     - `min_dist=0.4`: Controls point clustering density
     - `metric='cosine'`: Optimal for comparing semantic embeddings
     - `random_state=110`: Ensures reproducible results
     - `spread=2.0`: Creates clear separation between clusters
     - `n_epochs=750`: Higher quality embeddings with more training
     - `repulsion_strength=2.0`: Prevents overcrowding of clusters

5. **Visualisation**:
   - Interactive 3D plot using Plotly
   - Colour-coded by book category
   - Hover text shows book details
   - Optimised for clarity with 2400x1600 resolution

## Setup

1. Clone this repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Anthropic API key as an environment variable:
   ```bash
   export ANTHROPIC_API_KEY='your-key-here'
   ```

## Usage

Run the visualisation:
```bash
python bookvis.py
```

This will:
1. Fetch books from marksaroufim.com/bookshelf
2. Categorise them using Claude AI
3. Generate embeddings and create a 3D visualisation
4. Save the visualisation as an HTML file

## Output

The script generates an interactive 3D visualisation saved as an HTML file with timestamp (e.g., `book_visualization_umap_20240101_120000.html`). The visualisation allows you to:
- Rotate and zoom the 3D space
- Hover over points to see book details
- Toggle categories on/off in the legend
- Export the view as a PNG

## Technical Notes

- The script uses fixed random seeds (110) for reproducibility
- Embeddings are normalised using StandardScaler before UMAP
- Books are processed in chunks of 50 to avoid API limits
- The visualisation is optimised for larger screens
- UMAP parameters are tuned for:
  - Clear cluster separation
  - Meaningful local relationships
  - Visual clarity in 3D space
- SentenceTransformer parameters prioritise:
  - Efficient batch processing
  - Normalised embeddings for better similarity comparison
  - Adequate sequence length for book titles
