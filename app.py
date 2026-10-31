
import io
import textwrap

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from recommendation import MovieRecommender, download_image, prefetch_images

st.set_page_config(page_title="Movie Recommender", page_icon=":clapper:", layout="wide")



USE_REAL_POSTERS = True

errors = []          
COLORS = ["#E50914", "#1F2A44", "#C98A0B", "#3A4060", "#2F5DA8", "#7A1FA2", "#0B7A75", "#B5451B"]


def get_font(size):
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


@st.cache_data(show_spinner=False)
def make_poster(title, movie_id):
    """Draw a 300 x 450 poster (colour + year + wrapped title) and return PNG bytes."""
    name, year = (title.rsplit(" (", 1) + [""])[:2] if title.endswith(")") else (title, "")
    img = Image.new("RGB", (300, 450), COLORS[int(movie_id) % len(COLORS)])
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 300, 300, 450], fill="#101018")          # dark band at the bottom
    draw.text((20, 18), year.rstrip(")"), font=get_font(30), fill="white")
    y = 150
    for line in textwrap.wrap(name, width=14)[:5]:
        draw.text(((300 - draw.textlength(line, font=get_font(26))) / 2, y), line,
                  font=get_font(26), fill="white")
        y += 36
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def poster(title, movie_id, url):
   
    if USE_REAL_POSTERS:
        try:
            return download_image(url)
        except Exception as e:
            errors.append(str(e))
    return make_poster(title, int(movie_id))


@st.cache_resource(show_spinner="Loading model...")
def load_recommender():
    return MovieRecommender("model")


st.title("Personalized Movie Recommendation System")
st.caption("Item-based collaborative filtering: movies rated similarly by the same "
           "users are recommended together.")

try:
    rec = load_recommender()
except FileNotFoundError:
    st.error("Model files not found. Keep movies.pkl and similarity_score.pkl "
             "inside the 'model' folder.")
    st.stop()


st.sidebar.header("Settings")
n = st.sidebar.slider("Number of recommendations", 3, 10, 5)
st.sidebar.write(f"Movies available: {len(rec.titles)}")


selected = st.selectbox("Pick a movie you like (type to search)", rec.titles)

info = rec.get_movie(selected)
results = rec.recommend(selected, n)
if USE_REAL_POSTERS:
    prefetch_images([info["image_url"]] + results["image_url"].tolist())

left, right = st.columns([1, 4])

with left:
    st.subheader("You selected")
    st.image(poster(info["title"], info["movieId"], info["image_url"]))
    st.markdown(f"**{info['title']}**")

with right:
    st.subheader(f"Top {n} similar movies")
    per_row = 5
    for start in range(0, len(results), per_row):
        cols = st.columns(per_row)
        chunk = results.iloc[start:start + per_row]
        for col, (_, row) in zip(cols, chunk.iterrows()):
            with col:
                st.image(poster(row["title"], row["movieId"], row["image_url"]))
                st.markdown(f"**{row['title']}**")
                st.caption(f"Similarity: {row['similarity']:.2f}")

if errors:
    st.sidebar.warning(f"{len(errors)} poster(s) could not be downloaded, drawn posters "
                       f"are shown instead. First error: {errors[0][:150]}")
