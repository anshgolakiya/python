import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("movies_500_real_titles.csv")


# ============================================================
# FILL EMPTY VALUES
# ============================================================

features = [
    "genre",
    "keywords",
    "cast",
    "director",
    "description"
]

df[features] = df[features].fillna("")


# ============================================================
# COMBINE MOVIE FEATURES
# ============================================================

df["combined"] = df[features].agg(" ".join, axis=1)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["combined"]
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)


# ============================================================
# NORMALIZE MOVIE TITLES
# CASE-INSENSITIVE
# ============================================================

df["title_normalized"] = (
    df["title"]
    .astype(str)
    .str.strip()
    .str.casefold()
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# NORMALIZE INDUSTRY
# CASE-INSENSITIVE
# ============================================================

df["industry_normalized"] = (
    df["industry"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# CREATE MOVIE TITLE INDEX
# ============================================================

title_index = pd.Series(
    df.index,
    index=df["title_normalized"]
).drop_duplicates()


# ============================================================
# CREATE INDUSTRY LIST
# ============================================================

industries = df["industry_normalized"].unique()


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend(movie_name, count=15):

    # Convert count to integer
    count = int(count)

    # Make sure count is positive
    if count <= 0:
        count = 15

    # ========================================================
    # NORMALIZE USER INPUT
    # CASE-INSENSITIVE
    # ========================================================

    user_input = movie_name.strip().casefold()

    # ========================================================
    # CHECK IF USER ENTERED INDUSTRY
    # ========================================================

    if user_input in industries:

        industry_movies = df[
            df["industry_normalized"] == user_input
        ]

        # Randomly shuffle movies
        industry_movies = industry_movies.sample(
            frac=1
        )

        print("\n======================================")
        print("       MOVIES FROM INDUSTRY")
        print("======================================")

        print(
            "Industry:",
            industry_movies.iloc[0]["industry"]
        )

        print()

        # Display movies
        for number, (_, row) in enumerate(
            industry_movies.head(count).iterrows(),
            1
        ):

            print(
                f"{number}. {row['title']} "
                f"({row['industry']})"
            )

        return


    # ========================================================
    # CHECK IF USER ENTERED MOVIE
    # ========================================================

    if user_input not in title_index:

        print("\n======================================")
        print("          MOVIE NOT FOUND")
        print("======================================")

        print(
            "\nThe movie is not available "
            "in our dataset."
        )

        print(
            "\nHere are some random movies "
            "from our dataset:"
        )

        print("--------------------------------------")

        random_movies = df.sample(
            n=min(count, len(df))
        )

        for number, (_, row) in enumerate(
            random_movies.iterrows(),
            1
        ):

            print(
                f"{number}. {row['title']} "
                f"({row['industry']})"
            )

        return


    # ========================================================
    # GET SELECTED MOVIE INDEX
    # ========================================================

    index = title_index[user_input]


    # ========================================================
    # SHOW SELECTED MOVIE
    # ========================================================

    selected_movie = df.iloc[index]

    print("\n======================================")
    print("          SELECTED MOVIE")
    print("======================================")

    print(
        "Title       :",
        selected_movie["title"]
    )

    print(
        "Industry    :",
        selected_movie["industry"]
    )

    print(
        "Genre       :",
        selected_movie["genre"]
    )

    print(
        "Director    :",
        selected_movie["director"]
    )

    # Show description if available
    if str(selected_movie["description"]).strip():

        print(
            "Description :",
            selected_movie["description"]
        )


    # ========================================================
    # CALCULATE SIMILARITY
    # ========================================================

    similarity_scores = similarity_matrix[index]

    movie_indices = similarity_scores.argsort()[::-1]


    # ========================================================
    # SHOW RECOMMENDATIONS
    # ========================================================

    print("\n======================================")
    print("       RECOMMENDED MOVIES")
    print("======================================")

    shown = 0

    for i in movie_indices:

        # Do not recommend the selected movie
        if i == index:
            continue

        print(
            f"{shown + 1}. "
            f"{df.iloc[i]['title']} "
            f"(Industry: {df.iloc[i]['industry']}, "
            f"Similarity: {similarity_scores[i]:.2f})"
        )

        shown += 1

        if shown == count:
            break


# ============================================================
# MAIN PROGRAM
# ============================================================

print("======================================")
print("     MOVIE RECOMMENDATION SYSTEM")
print("======================================")

print("\nYou can enter:")

print("1. Movie name")
print("2. Industry name")

print("\nExamples:")

print("Avatar")
print("avatar")
print("AVATAR")
print("Bollywood")
print("Hollywood")
print("Marvel")
print("DC")
print("Animation")


# ============================================================
# USER INPUT
# ============================================================

movie = input(
    "\nEnter movie or industry name: "
)


# ============================================================
# CALL RECOMMENDATION FUNCTION
# ============================================================

recommend(movie, 15)
