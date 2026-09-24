import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("movies_500_actor_ready.csv")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "title",
    "genre",
    "keywords",
    "cast",
    "director",
    "description",
    "industry"
]

for column in required_columns:
    if column not in df.columns:
        print("Error: Column not found:", column)
        exit()


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
# CREATE COMBINED FEATURES
# ============================================================

df["combined"] = df[features].agg(" ".join, axis=1)


# ============================================================
# TF-IDF VECTORIZATION
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
# NORMALIZE TITLE
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
# ============================================================

df["industry_normalized"] = (
    df["industry"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# NORMALIZE CAST
# ============================================================

df["cast_normalized"] = (
    df["cast"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# NORMALIZE DIRECTOR
# ============================================================

df["director_normalized"] = (
    df["director"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# TITLE INDEX
# ============================================================

title_index = pd.Series(
    df.index,
    index=df["title_normalized"]
).drop_duplicates()


# ============================================================
# INDUSTRY LIST
# ============================================================

industries = set(
    df["industry_normalized"].unique()
)


# ============================================================
# ACTOR SEARCH
# ============================================================

def search_actor(actor_name, count):

    actor_input = actor_name.strip().casefold()

    result = df[
        df["cast_normalized"].str.contains(
            actor_input,
            regex=False,
            na=False
        )
    ]

    if len(result) == 0:
        return False

    print("\n======================================")
    print("           ACTOR MOVIES")
    print("======================================")

    print("Actor:", actor_name)
    print("Movies found:", len(result))

    print("--------------------------------------")

    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. {row['title']} "
            f"({row['industry']})"
        )

    return True


# ============================================================
# DIRECTOR SEARCH
# ============================================================

def search_director(director_name, count):

    director_input = director_name.strip().casefold()

    result = df[
        df["director_normalized"].str.contains(
            director_input,
            regex=False,
            na=False
        )
    ]

    if len(result) == 0:
        return False

    print("\n======================================")
    print("         DIRECTOR MOVIES")
    print("======================================")

    print("Director:", director_name)
    print("Movies found:", len(result))

    print("--------------------------------------")

    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. {row['title']} "
            f"({row['industry']})"
        )

    return True


# ============================================================
# INDUSTRY SEARCH
# ============================================================

def search_industry(industry_name, count):

    industry_input = industry_name.strip().casefold()

    result = df[
        df["industry_normalized"] == industry_input
    ]

    if len(result) == 0:
        return False

    result = result.sample(
        frac=1,
        random_state=None
    )

    print("\n======================================")
    print("          INDUSTRY MOVIES")
    print("======================================")

    print(
        "Industry:",
        result.iloc[0]["industry"]
    )

    print("Movies found:", len(result))

    print("--------------------------------------")

    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. {row['title']}"
        )

    return True


# ============================================================
# MOVIE RECOMMENDATION
# ============================================================

def recommend_movie(movie_name, count):

    movie_input = movie_name.strip().casefold()

    # Check movie
    if movie_input not in title_index:
        return False

    # Get movie index
    index = title_index[movie_input]

    # Get selected movie
    selected_movie = df.iloc[index]

    print("\n======================================")
    print("           SELECTED MOVIE")
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

    print(
        "Cast        :",
        selected_movie["cast"]
    )

    if str(selected_movie["description"]).strip():

        print(
            "Description :",
            selected_movie["description"]
        )


    # ========================================================
    # GET SIMILARITY SCORES
    # ========================================================

    similarity_scores = similarity_matrix[index]

    movie_indices = similarity_scores.argsort()[::-1]


    # ========================================================
    # DISPLAY RECOMMENDATIONS
    # ========================================================

    print("\n======================================")
    print("       RECOMMENDED MOVIES")
    print("======================================")

    shown = 0

    for i in movie_indices:

        # Do not show selected movie
        if i == index:
            continue

        print(
            f"{shown + 1}. "
            f"{df.iloc[i]['title']} "
            f"(Industry: {df.iloc[i]['industry']}, "
            f"Genre: {df.iloc[i]['genre']}, "
            f"Similarity: {similarity_scores[i]:.2f})"
        )

        shown += 1

        if shown == count:
            break

    return True


# ============================================================
# RANDOM MOVIES
# ============================================================

def random_movies(count):

    print("\n======================================")
    print("          RANDOM MOVIES")
    print("======================================")

    result = df.sample(
        n=min(count, len(df))
    )

    for number, (_, row) in enumerate(
        result.iterrows(),
        1
    ):

        print(
            f"{number}. {row['title']} "
            f"({row['industry']})"
        )


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend(user_input, count=15):

    user_input = str(user_input).strip()

    if user_input == "":
        print("\nPlease enter a movie, actor, director or industry.")
        return


    # ========================================================
    # 1. MOVIE
    # ========================================================

    if recommend_movie(user_input, count):
        return


    # ========================================================
    # 2. ACTOR
    # ========================================================

    if search_actor(user_input, count):
        return


    # ========================================================
    # 3. DIRECTOR
    # ========================================================

    if search_director(user_input, count):
        return


    # ========================================================
    # 4. INDUSTRY
    # ========================================================

    if search_industry(user_input, count):
        return


    # ========================================================
    # NOT FOUND
    # ========================================================

    print("\n======================================")
    print("             NOT FOUND")
    print("======================================")

    print(
        "\nMovie, actor, director or industry "
        "was not found in the dataset."
    )

    print("\nShowing random movies instead:")

    random_movies(count)


# ============================================================
# PROGRAM START
# ============================================================

print("======================================")
print("     MOVIE RECOMMENDATION SYSTEM")
print("======================================")

print("\nYou can search using:")

print("1. Movie name")
print("2. Actor name")
print("3. Director name")
print("4. Industry name")

print("\nExamples:")

print("Avatar")
print("Leonardo DiCaprio")
print("Robert Downey Jr")
print("Christopher Nolan")
print("Bollywood")
print("Hollywood")
print("Marvel")
print("DC")
print("Animation")


# ============================================================
# USER INPUT
# ============================================================

user_input = input(
    "\nEnter movie, actor, director or industry: "
)


# ============================================================
# NUMBER OF RECOMMENDATIONS
# ============================================================

count_input = input(
    "How many movies do you want? (default 15): "
)

if count_input.strip() == "":
    count = 15
else:
    try:
        count = int(count_input)

        if count <= 0:
            count = 15

    except ValueError:
        count = 15


# ============================================================
# RUN SYSTEM
# ============================================================

recommend(
    user_input,
    count
)