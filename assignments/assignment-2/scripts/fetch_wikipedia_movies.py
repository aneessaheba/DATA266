"""Fetch Wikipedia pages for 10 movies and save as .txt files for the RAG pipeline."""
import os
import wikipedia

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "wiki_movies")
os.makedirs(OUT_DIR, exist_ok=True)

MOVIES = {
    "Inception": "Inception",
    "The Matrix": "The Matrix",
    "Titanic (1997 film)": "Titanic_1997",
    "The Godfather": "The_Godfather",
    "Pulp Fiction": "Pulp_Fiction",
    "Forrest Gump": "Forrest_Gump",
    "The Dark Knight": "The_Dark_Knight",
    "Jurassic Park (film)": "Jurassic_Park",
    "The Shawshank Redemption": "The_Shawshank_Redemption",
    "Interstellar (film)": "Interstellar",
}

wikipedia.set_lang("en")

for query, fname in MOVIES.items():
    try:
        page = wikipedia.page(query, auto_suggest=False)
    except Exception as e:
        print(f"FAILED first try {query}: {e}, retrying with auto_suggest")
        page = wikipedia.page(query, auto_suggest=True)
    path = os.path.join(OUT_DIR, f"{fname}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Title: {page.title}\n\n")
        f.write(page.content)
    print(f"Saved {page.title!r} -> {path} ({len(page.content)} chars)")

print("Done. Files:", os.listdir(OUT_DIR))
