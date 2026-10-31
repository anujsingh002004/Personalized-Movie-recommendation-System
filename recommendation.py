
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_DIR = Path(__file__).parent
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MovieRecommender/1.0)"}

# One shared connection that automatically retries when the image site is busy (503 etc.)
session = requests.Session()
session.headers.update(HEADERS)
_retry = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
for _prefix in ("http://", "https://"):
    session.mount(_prefix, HTTPAdapter(max_retries=_retry))


class DownloadError(Exception):
    pass


_failed = {}            # url -> (time of failure, error text)
RETRY_AFTER = 60        # seconds to wait before trying a failed url again


@lru_cache(maxsize=1024)
def _download(url):
    response = session.get(url, timeout=10)
    response.raise_for_status()
    if not response.headers.get("Content-Type", "").startswith("image"):
        raise ValueError("URL did not return an image")
    return response.content


def download_image(url):
   
    failed = _failed.get(url)
    if failed and time.time() - failed[0] < RETRY_AFTER:
        raise DownloadError(failed[1])
    try:
        return _download(url)
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        _failed[url] = (time.time(), message)
        raise DownloadError(message) from e


def prefetch_images(urls):
    
    def safe_download(url):
        try:
            download_image(url)
        except Exception:
            pass                                  

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(safe_download, urls))


class MovieRecommender:
    def __init__(self, model_dir="model"):
        folder = BASE_DIR / model_dir

        # movies.pkl -> dict with lists: movieId, title, image_url
        with open(folder / "movies.pkl", "rb") as f:
            self.movies = pd.DataFrame(pickle.load(f))

        # similarity_score.pkl -> square matrix (movie x movie) of cosine similarity
        with open(folder / "similarity_score.pkl", "rb") as f:
            self.similarity = pickle.load(f)

        self.titles = self.movies["title"].tolist()
        self._position = {title: i for i, title in enumerate(self.titles)}

    def get_movie(self, title):
        """Return title, movieId and poster url of one movie."""
        return self.movies.iloc[self._position[title]].to_dict()

    def recommend(self, movie_name, n=5):
        """Return the n movies most similar to movie_name (best first)."""
        index = self._position[movie_name]
        scores = self.similarity[index]

        order = np.argsort(scores)[::-1]       # highest similarity first
        order = order[order != index][:n]      # remove the movie itself, keep top n

        result = self.movies.iloc[order].copy()
        result["similarity"] = scores[order]
        return result.reset_index(drop=True)
