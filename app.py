import streamlit as st
import pandas as pd
import requests
import os
from typing import List, Optional
from dotenv import load_dotenv

# ==================== CONFIG ====================
# Load environment variables from .env file
load_dotenv()

# API Configuration with fallback
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TOP_N = 10

# Data paths (relative to project root)
DATA_DIR = "tmdb_dataset"
MOVIES_FILE = os.path.join(DATA_DIR, "movies_final.csv")
RECOMMENDATIONS_FILE = os.path.join(DATA_DIR, "recommendations_top50.parquet")

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==================== CUSTOM CSS ====================
st.markdown(
    """
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .main {
        padding: 2rem;
    }

    /* Title */
    h1 {
        color: #ffffff !important;
        font-weight: 600;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Subheaders */
    h2, h3 {
        color: #f0f0f0 !important;
    }

    /* Recommendation items */
    .rec-item {
        border-bottom: 1px solid #444444;
        padding: 1rem 0;
    }

    .rec-item:last-child {
        border-bottom: none;
    }

    /* Movie title */
    .movie-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }

    /* Movie info text */
    .movie-info {
        font-size: 0.9rem;
        color: #cccccc;
        margin: 0.25rem 0;
    }

    /* Similarity score */
    .similarity-score {
        font-size: 0.95rem;
        font-weight: 500;
        color: #64b5f6;
    }

    /* Poster placeholder */
    .poster-placeholder {
        width: 150px;
        height: 225px;
        background-color: #2a2a2a;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 3rem;
        border-radius: 4px;
        border: 1px solid #444444;
    }

    /* Override Streamlit default text colors */
    .stMarkdown, .stText {
        color: #e0e0e0 !important;
    }

    /* Input fields */
    input {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 1px solid #555555 !important;
    }

    /* Select box */
    .st-emotion-cache-1gulkj5, .st-emotion-cache-16idsys {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
    }

    /* Button */
    .stButton > button {
        background-color: #1976d2 !important;
        color: #ffffff !important;
        border: none !important;
    }

    .stButton > button:hover {
        background-color: #1565c0 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== DATA LOADING ====================
@st.cache_data(show_spinner=False)
def load_data():
    """
    Load movies and recommendations data from CSV/Parquet files.

    Returns:
        tuple: (movies_df, recommendations_df)

    Raises:
        FileNotFoundError: If required data files are not found
    """
    try:
        movies_df = pd.read_csv(MOVIES_FILE)
        recommendations_df = pd.read_parquet(RECOMMENDATIONS_FILE)
        return movies_df, recommendations_df
    except FileNotFoundError as e:
        st.error(f"● Data file not found: {e}")
        st.info(
            "  Please run setup_notebook.ipynb first to generate required data files."
        )
        st.stop()


# ==================== HELPER FUNCTIONS ====================
@st.cache_data(ttl=3600, show_spinner=False)
def get_poster_url(tmdb_id: int) -> Optional[str]:
    """
    Fetch movie poster URL from TMDB API.

    Args:
        tmdb_id: TMDB movie ID

    Returns:
        Poster URL if found, None otherwise
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get("poster_path")
            if poster_path:
                return f"{TMDB_IMAGE_BASE_URL}{poster_path}"
    except Exception:
        pass
    return None


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text to specified length, breaking at word boundary.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Truncated text with ellipsis
    """
    if pd.isna(text):
        return "N/A"
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def get_movie_suggestions(
    query: str, movies_df: pd.DataFrame, limit: int = 5
) -> List[str]:
    """
    Get movie title suggestions based on search query.

    Args:
        query: Search query
        movies_df: Movies dataframe
        limit: Maximum number of suggestions

    Returns:
        List of matching movie titles
    """
    if len(query) < 3:
        return []
    mask = movies_df["title"].str.contains(query, case=False, na=False)
    return movies_df[mask]["title"].head(limit).tolist()


def display_poster(tmdb_id: int, width: int = 150):
    """
    Display movie poster or placeholder if not available.

    Args:
        tmdb_id: TMDB movie ID
        width: Poster width in pixels
    """
    poster_url = get_poster_url(tmdb_id)
    if poster_url:
        st.image(poster_url, width=width)
    else:
        st.markdown(
            f'<div class="poster-placeholder" style="width: {width}px; height: {int(width*1.5)}px;">🎬</div>',
            unsafe_allow_html=True,
        )


def format_value(value, default="N/A"):
    """
    Format value for display, returning default if missing.

    Args:
        value: Value to format
        default: Default value if missing

    Returns:
        Formatted value or default
    """
    return default if pd.isna(value) else value


def display_selected_movie(movie_row: pd.Series):
    """
    Display detailed information for selected movie.

    Args:
        movie_row: Pandas Series containing movie data
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        display_poster(movie_row["movie_id"], width=170)

    with col2:
        st.markdown(
            f"<div class='movie-title'>{movie_row['title']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='movie-info'><b>Genres:</b> {format_value(movie_row.get('genres_list'))}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='movie-info'><b>Runtime:</b> {format_value(movie_row.get('runtime_original'))} min</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='movie-info'><b>Release:</b> {format_value(movie_row.get('release_year'))}</div>",
            unsafe_allow_html=True,
        )

        vote_avg = format_value(movie_row.get("vote_average_original"))
        vote_cnt = format_value(movie_row.get("vote_count_original"))
        st.markdown(
            f"<div class='movie-info'><b>Rating:</b> {vote_avg} ({vote_cnt} votes)</div>",
            unsafe_allow_html=True,
        )

        overview = truncate_text(movie_row.get("overview"), 200)
        st.markdown(
            f"<div class='movie-info'><b>Overview:</b> {overview}</div>",
            unsafe_allow_html=True,
        )

        keywords = format_value(movie_row.get("keywords_list"))
        if keywords != "N/A":
            keywords = truncate_text(str(keywords), 100)
        st.markdown(
            f"<div class='movie-info'><b>Keywords:</b> {keywords}</div>",
            unsafe_allow_html=True,
        )


def display_recommendation_item(rank: int, rec_row: pd.Series):
    """
    Display a single recommendation item.

    Args:
        rank: Recommendation rank (1-10)
        rec_row: Pandas Series containing recommendation data
    """
    st.markdown('<div class="rec-item">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])

    with col1:
        display_poster(rec_row["recommended_id"], width=120)

    with col2:
        st.markdown(
            f"<div class='movie-title'>{rank}. {format_value(rec_row['title'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='similarity-score'>Similarity: {rec_row.get('similarity', 0):.3f}</div>",
            unsafe_allow_html=True,
        )

        runtime = format_value(rec_row.get("runtime_original"))
        rating = format_value(rec_row.get("vote_average_original"))
        st.markdown(
            f"<div class='movie-info'>Runtime: {runtime} min | Rating: {rating}</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<div class='movie-info'>Genres: {format_value(rec_row.get('genres_list'))}</div>",
            unsafe_allow_html=True,
        )

        overview = truncate_text(rec_row.get("overview"), 70)
        st.markdown(f"<div class='movie-info'>{overview}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def get_recommendations(
    movie_id: int,
    recommendations_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Get top-N recommendations for a given movie.

    Args:
        movie_id: Movie ID to get recommendations for
        recommendations_df: Pre-computed recommendations dataframe
        movies_df: Movies metadata dataframe
        top_n: Number of recommendations to return

    Returns:
        DataFrame containing top-N recommendations with full movie details
    """
    recs = recommendations_df[recommendations_df["movie_id"] == movie_id].head(top_n)
    if recs.empty:
        return pd.DataFrame()
    result = recs.merge(
        movies_df, left_on="recommended_id", right_on="movie_id", how="left"
    )
    return result.sort_values("rank")


# ==================== MAIN APP ====================
def main():
    """Main application logic."""

    st.title("Movie Recommendation System")
    st.markdown("---")

    # Load data
    with st.spinner("Loading movie database..."):
        movies_df, recommendations_df = load_data()

    # Show success message on first load
    if "loaded" not in st.session_state:
        st.success(
            f"✅ Loaded {len(movies_df):,} movies with pre-computed recommendations!"
        )
        st.session_state.loaded = True

    # Search interface
    st.subheader("Search for a movie")
    search_query = st.text_input(
        "Enter movie title",
        placeholder="Type at least 3 characters...",
        label_visibility="collapsed",
    )

    # Display suggestions based on search query
    if len(search_query) >= 3:
        suggestions = get_movie_suggestions(search_query, movies_df, limit=5)

        if suggestions:
            selected_title = st.selectbox(
                "Select a movie:", suggestions, key="movie_select"
            )
        else:
            st.warning("⚠️ No movies found. Try a different search term.")
            selected_title = None
    else:
        selected_title = None

    # Display selected movie and recommendations
    if selected_title:
        st.markdown("---")
        st.subheader("🎬 Selected Movie")

        movie_row = movies_df[movies_df["title"] == selected_title].iloc[0]
        display_selected_movie(movie_row)

        # Show recommendations button
        if st.button(
            "Show Recommendations", type="primary", use_container_width=True
        ):
            with st.spinner("Finding similar movies..."):
                recommendations = get_recommendations(
                    movie_row["movie_id"], recommendations_df, movies_df, top_n=TOP_N
                )

                if not recommendations.empty:
                    st.markdown("---")
                    st.subheader(f"⭐ Top {TOP_N} Similar Movies")

                    for idx, (_, rec_row) in enumerate(recommendations.iterrows(), 1):
                        display_recommendation_item(idx, rec_row)
                else:
                    st.error(
                        "❌ Could not generate recommendations for this movie. Please try another one."
                    )


if __name__ == "__main__":
    main()
