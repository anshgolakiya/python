import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("movies_500_real_titles.csv")


# ==============================
# FILL EMPTY VALUES
# ==============================

features = ["genre", "keywords", "cast", "director", "description"]

df[features] = df[features].fillna("")


# ==============================
# COMBINE MOVIE FEATURES
# ==============================

df["combined"] = df[features].agg(" ".join, axis=1)


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["combined"])


# ==============================
# COSINE SIMILARITY
# ==============================

similarity_matrix = cosine_similarity(tfidf_matrix)


# ==============================
# NORMALIZE MOVIE TITLES
# ==============================

df["title_normalized"] = (
    df["title"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
)


# ==============================
# NORMALIZE INDUSTRY
# ==============================

df["industry_normalized"] = (
    df["industry"]
    .astype(str)
    .str.strip()
    .str.lower()
)


# ==============================
# MOVIE TITLE INDEX
# ==============================

title_index = pd.Series(
    df.index,
    index=df["title_normalized"]
).drop_duplicates()


# ==============================
# INDUSTRY LIST
# ==============================

industries = df["industry_normalized"].unique()


# ==============================
# RECOMMEND FUNCTION
# ==============================

def recommend(movie_name, count=15):

    # Normalize user input
    user_input = movie_name.strip().lower()

    # --------------------------------
    # CHECK IF USER ENTERED INDUSTRY
    # --------------------------------

    if user_input in industries:

        industry_movies = df[
            df["industry_normalized"] == user_input
        ]

        # Randomly shuffle movies
        industry_movies = industry_movies.sample(
            frac=1,
            random_state=None
        )

        print("\nRecommended Movies")
        print("------------------")
        print("Industry:", user_input.title())

        print()

        # Display different movies
        for i, row in industry_movies.head(count).iterrows():

            print(
                f"{row['title']} "
                f"({row['industry']})"
            )

        return


    # --------------------------------
    # CHECK IF USER ENTERED MOVIE
    # --------------------------------

    if user_input not in title_index:

        print("\nMovie or Industry not found.")

        print("\nAvailable Industries:")

        for industry in sorted(industries):
            print("-", industry.title())

        print("\nExample:")
        print("Bollywood")
        print("Hollywood")
        print("Marvel")
        print("DC")
        print("Animation")

        return


    # --------------------------------
    # MOVIE RECOMMENDATION
    # --------------------------------

    index = title_index[user_input]

    similarity_scores = similarity_matrix[index]

    movie_indices = similarity_scores.argsort()[::-1]

    print("\nRecommended Movies")
    print("------------------")

    shown = 0

    for i in movie_indices:

        print(
            f"{df.iloc[i]['title']} "
            f"(Similarity: {similarity_scores[i]:.2f})"
        )

        shown += 1

        if shown == count:
            break


# ==============================
# MAIN PROGRAM
# ==============================

print("======================================")
print("     MOVIE RECOMMENDATION SYSTEM")
print("======================================")

print("\nYou can enter:")
print("1. Movie name")
print("2. Industry name")
print("\nExamples:")
print("Avatar")
print("Bollywood")
print("Hollywood")
print("Marvel")
print("DC")
print("Animation")

movie = input("\nEnter movie or industry name: ")

recommend(movie, 15)
