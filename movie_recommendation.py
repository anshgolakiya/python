import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("movies_100.csv")

features = ["genre", "keywords", "cast", "director", "description"]

df[features] = df[features].fillna("")

df["combined"] = df[features].agg(" ".join, axis=1)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["combined"])

similarity_matrix = cosine_similarity(tfidf_matrix)

title_index = pd.Series(
    df.index,
    index=df["title"].str.lower()
).drop_duplicates()


def recommend(movie_name, count=5):

    movie_name = movie_name.strip().lower()

    if movie_name not in title_index:
        print("Movie not found.")
        print("\nAvailable movies:")

        for title in df["title"].head(10):
            print("-", title)

        return

    index = title_index[movie_name]

    similarity_scores = similarity_matrix[index]

    movie_indices = similarity_scores.argsort()[::-1]

    print("\nRecommended Movies")
    print("------------------")

    shown = 0

    for i in movie_indices:

        if i == index:
            continue

        print(
            f"{df.iloc[i]['title']} "
            f"(Similarity: {similarity_scores[i]:.2f})"
        )

        shown += 1

        if shown == count:
            break


print("===== MOVIE RECOMMENDATION SYSTEM =====")

movie = input("Enter movie name: ")

recommend(movie, 5)