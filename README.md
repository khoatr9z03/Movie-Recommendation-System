# Movie Recommendation System

A content-based filtering movie recommendation system using cosine similarity. Built with Python, Streamlit, and TMDB API.

## Overview

This project implements a **Content-Based Filtering** recommendation system for movies using:
- **TF-IDF** for text features (overview, keywords)
- **Multi-hot encoding** for genres
- **One-hot encoding** for languages
- **Standardized numeric features** (runtime, ratings, popularity)
- **Cosine similarity** for computing movie similarities

The system recommends movies based on content features rather than user ratings.


## Features

- **Search movies** by title with autocomplete
- **Get top-N recommendations** based on content similarity
- **Display movie posters** fetched from TMDB API
- **Show detailed movie info** (genres, runtime, ratings, overview)
- **Fast loading** with pre-computed recommendations (Parquet format)
- **16,000+ movies** in the database

## Project Structure

```
movie-recommendation-system/
├── app.py                      # Streamlit web application
├── setup_notebook.ipynb        # Setup and data verification notebook
├── MovieRecSys.ipynb           # Main pipeline (collection → modeling)
├── requirements.txt            # Python dependencies
├── .env                        # API keys (included for demo)
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── tmdb_dataset/               # Data directory
    ├── movies_final.csv                # Movie metadata (16,286 movies)
    ├── recommendations_top50.parquet   # Pre-computed recommendations
    ├── similarity_matrix.npy           # Cosine similarity matrix
    ├── movie_indices.csv               # Movie ID mappings
    ├── genre_mlb.pkl                   # Genre encoder
    ├── tfidf_keywords.pkl              # Keywords vectorizer
    └── tfidf_overview.pkl              # Overview vectorizer
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/khoatr9z03/movie-recommendation-system.git
cd movie-recommendation-system
```

Or download as ZIP and extract.

### Step 2: Download the dataset
- **Download:** https://drive.google.com/drive/folders/1LS3q_rY4UD-hhVT5dC6_c2A6DNQN5GsZ?usp=sharing
- Then move the dataset folder to this folder (MOVIE RECOMMENDATION SYSTEM)

### Step 3: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import streamlit; import pandas; import numpy; print('All packages installed!')"
```


## Usage

### Quick Start

1. **Run setup notebook** (Do this if you want to regenerate recommendations, the file "recommendations_top50" already existed as default):
   ```bash
   jupyter notebook setup_notebook.ipynb
   ```
   - This generates `recommendations_top50.parquet` from `similarity_matrix.npy`
   - Verifies all data files are present

2. **Launch Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open in browser**:
   - App will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in terminal

### Using the App

1. **Search for a movie**:
   - Type at least 3 characters in the search box
   - Select from autocomplete suggestions

2. **View movie details**:
   - See poster, genres, runtime, ratings, overview, and keywords

3. **Get recommendations**:
   - Click "Show Recommendations" button
   - See top 10 similar movies with similarity scores

4. **Explore recommendations**:
   - Each recommendation shows poster, genres, ratings, and overview
   - Similarity scores range from 0 (not similar) to 1 (identical)


## Data Pipeline

The complete pipeline (in `MovieRecSys.ipynb`):

### 1. Data Collection
- Source: TMDB API
- Method: Stratified sampling across popularity tiers
- Size: 23,000+ movies initially

### 2. Data Preprocessing
- **Missing values**: Drop/fill based on importance
- **Outliers**: IQR method with conditional handling
- **Tiering**: Quality-based filtering (TIER1/TIER2/TIER3)
- Final dataset: 16,286 movies

### 3. Feature Engineering
- **Text features**: TF-IDF for overview (400 features) and keywords (200 features)
- **Categorical**: Multi-hot for genres (19 features), One-hot for languages (71 features)
- **Numeric**: Standardized features (runtime, ratings, popularity, age, etc.)
- **Total**: 698 features

### 4. Modeling
- Content-Based Filtering
- **Similarity**: Cosine similarity
- **Feature weighting**: Genres (4.0x), Keywords (2.5x), Overview (1.5x)
- **Optimization**: Pre-computed top-50 recommendations per movie

### 5. Evaluation
- **Genre overlap**: 99% 
- **Diversity**: 12% 
- **Coverage**: 6.25% 


## Technologies

### Core
- **Python 3.8+**: Programming language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: TF-IDF, cosine similarity

### Web App
- **Streamlit**: Interactive web interface
- **Requests**: TMDB API calls
- **PyArrow**: Parquet file support

### Data Processing
- **TF-IDF Vectorization**: Text feature extraction
- **StandardScaler**: Numeric feature normalization
- **MultiLabelBinarizer**: Genre encoding

### External APIs
- **TMDB API**: Movie posters and metadata


## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total Movies | 16,286 |
| Total Features | 698 |
| Genres | 19 unique |
| Languages | 71 unique |
| Overview | 400 |
| Keyword | 200 |
| Numeric features | 8 |
