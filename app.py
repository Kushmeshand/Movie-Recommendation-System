import streamlit as st
import requests
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import io
import sqlite3

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
conn = sqlite3.connect("movies.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    movie_title TEXT
)
""")

conn.commit()

# ─────────────────────────────────────────────
# GLOBAL THEME
# ─────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
    --red:      #e50914;
    --red-dim:  #9b0610;
    --gold:     #f5c518;
    --bg:       #0a0a0f;
    --surface:  #111118;
    --surface2: #1a1a24;
    --border:   rgba(255,255,255,0.07);
    --text:     #f0eff4;
    --muted:    #8a8a99;
    --font-display: 'Bebas Neue', sans-serif;
    --font-body:    'DM Sans', sans-serif;
}

.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(229,9,20,0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(229,9,20,0.05), transparent);
    font-family: var(--font-body);
}



# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def section_header(label, emoji=""):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin:2rem 0 1rem;">
        <span style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;
                     letter-spacing:0.08em;color:#f0eff4;">{emoji} {label}</span>
        <div style="flex:1;height:1px;background:rgba(255,255,255,0.07);"></div>
    </div>""", unsafe_allow_html=True)

def movie_card(title, poster, score=None, rating=None):
    badge = ""
    if score is not None:
        badge = f"""<div style="position:absolute;top:8px;right:8px;
            background:rgba(229,9,20,0.92);color:white;font-size:11px;
            font-weight:600;padding:3px 8px;border-radius:4px;
            font-family:'DM Sans',sans-serif;">{score}%</div>"""
    elif rating is not None:
        badge = f"""<div style="position:absolute;top:8px;right:8px;
            background:rgba(245,197,24,0.92);color:#0a0a0f;font-size:11px;
            font-weight:700;padding:3px 8px;border-radius:4px;
            font-family:'DM Sans',sans-serif;">★ {rating}</div>"""

    short = title if len(title) <= 22 else title[:20] + "…"
    st.markdown(f"""
    <div style="position:relative;margin-bottom:4px;">
        <img src="{poster}" style="width:100%;border-radius:8px;display:block;
             transition:transform 0.22s ease,box-shadow 0.22s ease;"
             onmouseover="this.style.transform='scale(1.04) translateY(-3px)';
                          this.style.boxShadow='0 12px 32px rgba(0,0,0,0.7)'"
             onmouseout="this.style.transform='scale(1)';
                         this.style.boxShadow='none'"/>
        {badge}
    </div>
    <p style="font-family:'DM Sans',sans-serif;font-size:13px;font-weight:500;
              color:#f0eff4;margin:4px 0 6px;line-height:1.3;">{short}</p>
    """, unsafe_allow_html=True)

def watchlist_item(title):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 14px;
        background:#111118;border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;margin-bottom:6px;font-family:'DM Sans',sans-serif;
        font-size:14px;color:#f0eff4;">
        <span style="color:#e50914;font-size:14px;">▶</span>{title}
    </div>""", unsafe_allow_html=True)

def detail_stat(label, value):
    st.markdown(f"""
    <div style="background:#111118;border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:12px 14px;margin-bottom:6px;">
        <div style="font-size:11px;color:#8a8a99;font-family:'DM Sans',sans-serif;
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">{label}</div>
        <div style="font-size:16px;font-weight:500;color:#f0eff4;
                    font-family:'DM Sans',sans-serif;">{value}</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD PKL FROM DRIVE
# ─────────────────────────────────────────────
def load_pkl_from_drive(file_id):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            response = session.get(URL, params={"id": file_id, "confirm": value}, stream=True)
            break
    return pickle.load(io.BytesIO(response.content))

@st.cache_resource
def load_collab_data():
    sim    = load_pkl_from_drive("1hLX4egbrSd4hG2Qf1jhB5x5SumqtqO5V")
    movies = load_pkl_from_drive("15XF2K6hnPOHJG8e3RI-jnyXdxci43yrU")
    return sim, movies

collab_similarity, collab_movies = load_collab_data()

API_KEY  = "4e4b10932b7c2b31fd1e0a074c80f0c9"
OMDB_KEY = "e15bce82"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clean_title(title):
    import re
    title = re.sub(r"\(.*?\)", "", title)
    title = re.sub(r"a\.k\.a.*", "", title, flags=re.IGNORECASE)
    if ", The" in title:
        title = "The " + title.replace(", The", "")
    return title.strip()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, default in [
    ("selected_movie_details", None),
    ("watchlist", []),
    ("user", None),
    ("watchlists", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────
# API CALLS
# ─────────────────────────────────────────────
def fetch_poster(movie_title):
    try:
        cleaned = clean_title(movie_title)
        url  = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned}"
        data = requests.get(url).json()
        if data["results"]:
            poster = data["results"][0].get("poster_path")
            if poster:
                return "https://image.tmdb.org/t/p/w500" + poster
    except Exception:
        pass
    return "https://via.placeholder.com/300x450.png?text=No+Image"

def fetch_trending():
    try:
        url  = f"https://api.themoviedb.org/3/trending/movie/day?api_key={API_KEY}"
        data = requests.get(url).json()
        movies = []
        for m in data["results"][:5]:
            poster = "https://image.tmdb.org/t/p/w500" + m["poster_path"] if m.get("poster_path") else ""
            movies.append({
                "title":  m["title"],
                "poster": poster,
                "rating": round(m["vote_average"], 1)
            })
        return movies
    except Exception:
        return []

def fetch_details(movie_title):
    try:
        import concurrent.futures
        cleaned = clean_title(movie_title)
        search  = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={cleaned}"
        data    = requests.get(search).json()
        if not data["results"]:
            return None
        movie    = data["results"][0]
        movie_id = movie["id"]
        overview = movie.get("overview")

        with concurrent.futures.ThreadPoolExecutor() as ex:
            f_info    = ex.submit(requests.get, f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}")
            f_videos  = ex.submit(requests.get, f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}")
            f_credits = ex.submit(requests.get, f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={API_KEY}")
            f_omdb    = ex.submit(requests.get, f"http://www.omdbapi.com/?t={cleaned}&apikey={OMDB_KEY}")

        movie_info = f_info.result().json()
        videos     = f_videos.result().json()
        credits    = f_credits.result().json()
        omdb       = f_omdb.result().json()

        runtime      = movie_info.get("runtime")
        release_date = movie_info.get("release_date")
        rating       = movie_info.get("vote_average")
        popularity   = movie_info.get("popularity")
        votes        = movie_info.get("vote_count")

        trailer = None
        for v in videos["results"]:
            if v["type"] == "Trailer":
                trailer = v["key"]
                break

        cast     = credits["cast"][:5]
        director = next((c for c in credits["crew"] if c["job"] == "Director"), None)

        rt_rating = "N/A"
        if omdb.get("Ratings"):
            for r in omdb["Ratings"]:
                if r["Source"] == "Rotten Tomatoes":
                    rt_rating = r["Value"]

        return {
            "trailer": trailer, "cast": cast, "director": director,
            "runtime": runtime, "release_date": release_date,
            "rating": rating, "rt": rt_rating,
            "popularity": popularity, "votes": votes, "overview": overview
        }
    except Exception:
        return None

# ─────────────────────────────────────────────
# DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    movies  = pd.read_csv("https://drive.google.com/uc?export=download&id=1MvT-iG8f837YFmZdXzcuuhhfh4WYTZQq")
    credits = pd.read_csv("https://drive.google.com/uc?export=download&id=15EBSjEpdVoSrtPRQIuv_fvxlbgQeSuTZ")
    return movies, credits

@st.cache_data
def preprocess():
    movies, credits = load_data()
    movies = movies.merge(credits, on="title")
    movies = movies[['id','title','overview','genres','keywords','cast','crew','vote_average']]
    movies = movies.dropna()

    def convert(text):       return [i['name'] for i in ast.literal_eval(text)]
    def convert_cast(text):  return [i['name'] for i in ast.literal_eval(text)[:3]]
    def fetch_director(text):
        for i in ast.literal_eval(text):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    movies['genres']   = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast']     = movies['cast'].apply(convert_cast)
    movies['crew']     = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    GENRE_WEIGHT   = 3
    KEYWORD_WEIGHT = 2
    movies['tags'] = (movies['overview']
                      + movies['genres']   * GENRE_WEIGHT
                      + movies['keywords'] * KEYWORD_WEIGHT
                      + movies['cast']
                      + movies['crew'])

    new_df = movies[['id','title','tags','overview','vote_average']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())
    return new_df

@st.cache_data
def create_model(df):
    cv      = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
    return cosine_similarity(vectors)

new_df     = preprocess()
similarity = create_model(new_df)

# ─────────────────────────────────────────────
# HYBRID RECOMMENDER
# ─────────────────────────────────────────────
def hybrid_recommend(movie):
    matches = new_df[new_df['title'] == movie]
    if matches.empty:
        st.error(f"'{movie}' not found in dataset.")
        return []
    index     = matches.index[0]
    distances = similarity[index]

    content_scores = {new_df.iloc[i].title: score for i, score in enumerate(distances)}

    collab_scores = {}
    match = next((m for m in collab_movies if movie.lower() in m.lower()), None)
    if match:
        idx = collab_movies.index(match)
        for i, score in enumerate(collab_similarity[idx]):
            collab_scores[collab_movies[i]] = score

    max_content = max(content_scores.values()) or 1
    max_collab  = max(collab_scores.values())  if collab_scores else 1

    final_scores = {}
    for title in content_scores:
        nc = content_scores[title] / max_content
        cc = collab_scores.get(title, 0) / max_collab
        final_scores[title] = 0.6 * nc + 0.4 * cc

    top = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[1:6]
    return [{"title": t, "poster": fetch_poster(t), "score": round(s * 100, 1)} for t, s in top]

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3rem 0 2rem;position:relative;">
    <div style="font-family:'Bebas Neue',sans-serif;
                font-size:clamp(56px,10vw,96px);letter-spacing:0.06em;
                line-height:1;color:#f0eff4;margin-bottom:4px;">
        CINE<span style="color:#e50914;">MATCH</span>
        <span style="font-size:0.35em;vertical-align:super;
                     color:#f5c518;letter-spacing:0.12em;">AI</span>
    </div>
    <p style="font-family:'DM Sans',sans-serif;font-size:14px;
              letter-spacing:0.22em;color:#8a8a99;
              text-transform:uppercase;margin:0;">
        Your personalised movie universe
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOGIN / SIGNUP
# ─────────────────────────────────────────────
if st.session_state.user is None:
    st.markdown("""
    <div style="background:#111118;border:1px solid rgba(255,255,255,0.08);
        border-radius:14px;padding:2rem 2rem 0.5rem;
        max-width:400px;margin:1rem auto 0;">
        <p style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;
                  letter-spacing:0.08em;color:#f0eff4;margin:0 0 1.2rem;
                  text-align:center;">WELCOME BACK</p>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        username = st.text_input("Username", label_visibility="visible")
        password = st.text_input("Password", type="password", label_visibility="visible")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login", use_container_width=True):
                cursor.execute(
                    "SELECT * FROM users WHERE username=? AND password=?",
                    (username, password)
                )
                if cursor.fetchone():
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        with c2:
            if st.button("Sign up", use_container_width=True):
                try:
                    cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
                    conn.commit()
                    st.success("Account created! Please log in.")
                except Exception:
                    st.error("Username already exists")

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# WELCOME BAR + LOGOUT
# ─────────────────────────────────────────────
col_wel, col_out = st.columns([5, 1])
with col_wel:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:10px 16px;
        background:#111118;border:1px solid rgba(255,255,255,0.06);
        border-radius:10px;margin-bottom:1rem;">
        <div style="width:32px;height:32px;border-radius:50%;
            background:rgba(229,9,20,0.18);display:flex;align-items:center;
            justify-content:center;font-family:'Bebas Neue',sans-serif;
            font-size:16px;color:#e50914;letter-spacing:0.05em;">
            {st.session_state.user[0].upper()}
        </div>
        <span style="font-size:14px;color:#8a8a99;font-family:'DM Sans',sans-serif;">
            Signed in as <strong style="color:#f0eff4;">{st.session_state.user}</strong>
        </span>
    </div>""", unsafe_allow_html=True)
with col_out:
    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

# ─────────────────────────────────────────────
# TRENDING
# ─────────────────────────────────────────────
section_header("Trending Now", "🔥")

with st.spinner("Loading trending movies…"):
    trending = fetch_trending()

cols = st.columns(5)
for i, movie in enumerate(trending):
    with cols[i]:
        movie_card(movie["title"], movie["poster"], rating=movie["rating"])
        if st.button("View Details", key=f"trend_{i}"):
            st.session_state.selected_movie_details = movie

# ─────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────
section_header("My Watchlist", "❤️")

cursor.execute("SELECT movie_title FROM watchlist WHERE username=?", (st.session_state.user,))
items = cursor.fetchall()

if items:
    wl_cols = st.columns(2)
    for idx, item in enumerate(items):
        with wl_cols[idx % 2]:
            c_title, c_rm = st.columns([5, 1])
            with c_title:
                watchlist_item(item[0])
            with c_rm:
                if st.button("✕", key=f"rm_{item[0]}"):
                    cursor.execute(
                        "DELETE FROM watchlist WHERE username=? AND movie_title=?",
                        (st.session_state.user, item[0])
                    )
                    conn.commit()
                    st.rerun()
else:
    st.markdown("""
    <div style="padding:20px;text-align:center;
        border:1px dashed rgba(255,255,255,0.1);border-radius:10px;">
        <p style="color:#8a8a99;font-size:14px;margin:0;">
            No movies saved yet — add some from the recommendations below!
        </p>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SEARCH & RECOMMEND
# ─────────────────────────────────────────────
section_header("Find Your Next Watch", "🎬")

search_query = st.text_input("Search a movie title", placeholder="e.g. Inception, The Dark Knight…")
if search_query:
    filtered_list = [t for t in new_df['title'] if search_query.lower() in t.lower()]
    movie_list    = filtered_list[:60] if filtered_list else ["No results found"]
else:
    movie_list = ["— Select a movie —"] + sorted(list(new_df['title']))

selected_movie = st.selectbox("Select a movie", movie_list, label_visibility="collapsed")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if st.button("✦  Get Recommendations", use_container_width=True):
    valid = selected_movie not in ["— Select a movie —", "No results found"]
    if valid:
        with st.spinner("Finding the best matches…"):
            st.session_state.recommendations = hybrid_recommend(selected_movie)
    else:
        st.warning("Please select a movie first.")

# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────
if "recommendations" in st.session_state and st.session_state.recommendations:
    section_header("Recommended For You", "🎯")

    rec_cols = st.columns(5)
    for i, movie in enumerate(st.session_state.recommendations):
        with rec_cols[i]:
            movie_card(movie["title"], movie["poster"], score=movie["score"])

            if st.button("Details", key=f"rec_det_{i}"):
                st.session_state.selected_movie_details = movie

            if st.session_state.user:
                if st.button("❤ Save", key=f"watch_{i}"):
                    cursor.execute(
                        "SELECT * FROM watchlist WHERE username=? AND movie_title=?",
                        (st.session_state.user, movie["title"])
                    )
                    if not cursor.fetchone():
                        cursor.execute(
                            "INSERT INTO watchlist (username, movie_title) VALUES (?, ?)",
                            (st.session_state.user, movie["title"])
                        )
                        conn.commit()
                        st.success(f"Added '{movie['title']}'")
                        st.rerun()
                    else:
                        st.info("Already in watchlist")

# ─────────────────────────────────────────────
# MOVIE DETAILS
# ─────────────────────────────────────────────
if st.session_state.selected_movie_details:
    movie   = st.session_state.selected_movie_details

    st.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:2.5rem 0 1.5rem;'>",
                unsafe_allow_html=True)

    with st.spinner(f"Loading details for {movie['title']}…"):
        details = fetch_details(movie["title"])

    # Title
    st.markdown(f"""
    <h1 style="font-family:'Bebas Neue',sans-serif;font-size:clamp(32px,5vw,56px);
               letter-spacing:0.05em;color:#f0eff4;margin:0 0 1.2rem;">
        {movie['title']}
    </h1>""", unsafe_allow_html=True)

    if details:
        # Stats row
        stat_cols = st.columns(4)
        with stat_cols[0]: detail_stat("IMDb Rating", f"⭐ {details['rating']}")
        with stat_cols[1]: detail_stat("Rotten Tomatoes", f"🍅 {details['rt']}")
        with stat_cols[2]: detail_stat("Runtime", f"{details['runtime']} min")
        with stat_cols[3]: detail_stat("Release", str(details['release_date'])[:4])

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Overview
        if details.get("overview"):
            st.markdown(f"""
            <div style="background:#111118;border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:16px 20px;margin:0.8rem 0;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:1rem;
                            letter-spacing:0.1em;color:#8a8a99;margin-bottom:8px;">
                    OVERVIEW
                </div>
                <p style="font-family:'DM Sans',sans-serif;font-size:15px;
                          line-height:1.7;color:#d0cfd8;margin:0;">
                    {details['overview']}
                </p>
            </div>""", unsafe_allow_html=True)

        # Trailer
        if details.get("trailer"):
            section_header("Trailer", "▶")
            st.video(f"https://www.youtube.com/watch?v={details['trailer']}")

        # Director
        if details.get("director"):
            section_header("Director", "🎥")
            director = details["director"]
            d_col1, d_col2 = st.columns([1, 5])
            with d_col1:
                if director.get("profile_path"):
                    st.image("https://image.tmdb.org/t/p/w200" + director["profile_path"],
                             use_container_width=True)
            with d_col2:
                st.markdown(f"""
                <p style="font-family:'DM Sans',sans-serif;font-size:16px;
                           font-weight:500;color:#f0eff4;margin:12px 0 0;">
                    {director['name']}
                </p>""", unsafe_allow_html=True)

        # Cast
        if details.get("cast"):
            section_header("Cast", "👥")
            cast_cols = st.columns(5)
            for i, actor in enumerate(details["cast"]):
                with cast_cols[i]:
                    if actor.get("profile_path"):
                        st.image("https://image.tmdb.org/t/p/w200" + actor["profile_path"],
                                 use_container_width=True)
                    st.markdown(f"""
                    <p style="font-family:'DM Sans',sans-serif;font-size:12px;
                               font-weight:500;color:#f0eff4;margin:4px 0 0;
                               text-align:center;">
                        {actor['name']}
                    </p>""", unsafe_allow_html=True)

    # Close button
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("✕  Close Details"):
        st.session_state.selected_movie_details = None
        st.rerun()
