import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz


# ============================================================
# LOAD CSV
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

        print("ERROR: Column not found:", column)
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
    .str.replace(r"\s+", " ", regex=True)
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
    .str.replace(r"\s+", " ", regex=True)
)


# ============================================================
# NORMALIZE GENRE
# ============================================================

df["genre_normalized"] = (
    df["genre"]
    .astype(str)
    .str.strip()
    .str.casefold()
)


# ============================================================
# COMBINE FEATURES
# ============================================================

df["combined"] = (
    df["genre"] + " " +
    df["keywords"] + " " +
    df["cast"] + " " +
    df["director"] + " " +
    df["description"]
)


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
# MOVIE TITLE INDEX
# ============================================================

title_index = {}

for index, title in enumerate(
    df["title_normalized"]
):

    if title not in title_index:

        title_index[title] = index


# ============================================================
# INDUSTRY LIST
# ============================================================

industries = set(
    df["industry_normalized"]
)


# ============================================================
# CREATE ACTOR LIST
# ============================================================

all_actors = set()

for cast in df["cast_normalized"]:

    actors = cast.split("|")

    for actor in actors:

        actor = actor.strip()

        if actor:

            all_actors.add(actor)


# ============================================================
# ACTOR SEARCH
# 75% FUZZY MATCH
# ============================================================

def search_actor(actor_name, count):

    actor_input = (
        actor_name
        .strip()
        .casefold()
    )

    actor_input = " ".join(
        actor_input.split()
    )


    # --------------------------------------------------------
    # FIND BEST ACTOR MATCH
    # --------------------------------------------------------

    best_actor = None
    best_score = 0


    for actor in all_actors:

        score = fuzz.ratio(
            actor_input,
            actor
        )

        if score > best_score:

            best_score = score
            best_actor = actor


    # --------------------------------------------------------
    # REQUIRE 75% MATCH
    # --------------------------------------------------------

    if best_score < 75:

        return False


    # --------------------------------------------------------
    # FIND ACTOR MOVIES
    # --------------------------------------------------------

    result = df[
        df["cast_normalized"].apply(
            lambda cast:
            best_actor in [
                actor.strip()
                for actor in cast.split("|")
            ]
        )
    ]


    if len(result) == 0:

        return False


    # --------------------------------------------------------
    # DISPLAY ACTOR RESULT
    # --------------------------------------------------------

    print("\n======================================")
    print("       ACTOR MOVIE RECOMMENDATIONS")
    print("======================================")

    print(
        "Actor entered :",
        actor_name
    )

    print(
        "Matched actor :",
        best_actor.title()
    )

    print(
        "Match score   :",
        round(best_score, 2),
        "%"
    )

    print(
        "Movies found  :",
        len(result)
    )

    print("--------------------------------------")


    # --------------------------------------------------------
    # DISPLAY MOVIES
    # --------------------------------------------------------

    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. "
            f"{row['title']} "
            f"({row['industry']})"
        )


    return True


# ============================================================
# DIRECTOR SEARCH
# ============================================================

def search_director(
    director_name,
    count
):

    director_input = (
        director_name
        .strip()
        .casefold()
    )


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

    print(
        "Director:",
        director_name
    )

    print(
        "Movies found:",
        len(result)
    )

    print("--------------------------------------")


    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. "
            f"{row['title']} "
            f"({row['industry']})"
        )


    return True


# ============================================================
# INDUSTRY SEARCH
# ============================================================

def search_industry(
    industry_name,
    count
):

    industry_input = (
        industry_name
        .strip()
        .casefold()
    )


    result = df[
        df["industry_normalized"]
        == industry_input
    ]


    if len(result) == 0:

        return False


    # Random order
    result = result.sample(
        frac=1
    )


    print("\n======================================")
    print("          INDUSTRY MOVIES")
    print("======================================")

    print(
        "Industry:",
        result.iloc[0]["industry"]
    )

    print(
        "Movies found:",
        len(result)
    )

    print("--------------------------------------")


    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. "
            f"{row['title']}"
        )


    return True


# ============================================================
# GENRE / MOVIE TYPE SEARCH
# ============================================================

def search_genre(
    genre_name,
    count
):

    genre_input = (
        genre_name
        .strip()
        .casefold()
    )


    # --------------------------------------------------------
    # FIND GENRE
    # --------------------------------------------------------

    result = df[
        df["genre_normalized"].str.contains(
            genre_input,
            regex=False,
            na=False
        )
    ]


    # --------------------------------------------------------
    # GENRE NOT FOUND
    # --------------------------------------------------------

    if len(result) == 0:

        print("\n======================================")
        print("          GENRE NOT FOUND")
        print("======================================")

        print(
            f"\n{genre_name} type movie is not in dataset."
        )

        print(
            "\nShowing random movies instead:"
        )

        random_movies(count)

        return True


    # --------------------------------------------------------
    # GENRE FOUND
    # --------------------------------------------------------

    print("\n======================================")
    print("          GENRE MOVIES")
    print("======================================")

    print(
        "Movie Type:",
        genre_name
    )

    print(
        "Movies found:",
        len(result)
    )

    print("--------------------------------------")


    for number, (_, row) in enumerate(
        result.head(count).iterrows(),
        1
    ):

        print(
            f"{number}. "
            f"{row['title']} "
            f"(Genre: {row['genre']}, "
            f"Industry: {row['industry']})"
        )


    return True


# ============================================================
# MOVIE RECOMMENDATION
# ============================================================

def recommend_movie(
    movie_name,
    count
):

    movie_input = (
        movie_name
        .strip()
        .casefold()
    )


    # --------------------------------------------------------
    # CHECK MOVIE
    # --------------------------------------------------------

    if movie_input not in title_index:

        return False


    # --------------------------------------------------------
    # GET MOVIE INDEX
    # --------------------------------------------------------

    index = title_index[
        movie_input
    ]


    # --------------------------------------------------------
    # SELECTED MOVIE
    # --------------------------------------------------------

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


    if str(
        selected_movie["description"]
    ).strip():

        print(
            "Description :",
            selected_movie["description"]
        )


    # --------------------------------------------------------
    # GET SIMILARITY SCORES
    # --------------------------------------------------------

    similarity_scores = (
        similarity_matrix[index]
    )


    movie_indices = (
        similarity_scores
        .argsort()[::-1]
    )


    # --------------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # --------------------------------------------------------

    print("\n======================================")
    print("       RECOMMENDED MOVIES")
    print("======================================")


    shown = 0


    for i in movie_indices:

        # Don't recommend same movie
        if i == index:

            continue


        print(
            f"{shown + 1}. "
            f"{df.iloc[i]['title']} "
            f"(Industry: "
            f"{df.iloc[i]['industry']}, "
            f"Genre: "
            f"{df.iloc[i]['genre']}, "
            f"Similarity: "
            f"{similarity_scores[i]:.2f})"
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
        n=min(
            count,
            len(df)
        )
    )


    for number, (_, row) in enumerate(
        result.iterrows(),
        1
    ):

        print(
            f"{number}. "
            f"{row['title']} "
            f"({row['industry']})"
        )


# ============================================================
# MAIN RECOMMENDATION FUNCTION
# ============================================================

def recommend(
    user_input,
    count=15
):

    user_input = str(
        user_input
    ).strip()


    if user_input == "":

        print(
            "\nPlease enter a movie, actor, "
            "director, industry or genre."
        )

        return


    # ========================================================
    # 1. MOVIE
    # ========================================================

    if recommend_movie(
        user_input,
        count
    ):

        return


    # ========================================================
    # 2. ACTOR
    # ========================================================

    if search_actor(
        user_input,
        count
    ):

        return


    # ========================================================
    # 3. DIRECTOR
    # ========================================================

    if search_director(
        user_input,
        count
    ):

        return


    # ========================================================
    # 4. INDUSTRY
    # ========================================================

    if search_industry(
        user_input,
        count
    ):

        return


    # ========================================================
    # 5. GENRE / MOVIE TYPE
    # ========================================================

    if search_genre(
        user_input,
        count
    ):

        return


    # ========================================================
    # 6. NOT FOUND
    # ========================================================

    print("\n======================================")
    print("             NOT FOUND")
    print("======================================")

    print(
        "\nMovie, actor, director, industry "
        "or genre was not found."
    )

    print(
        "\nShowing random movies instead:"
    )

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
print("5. Genre / Movie type")


print("\nExamples:")

print("Avatar")
print("Robert Downey Jr")
print("Leonardo DiCaprio")
print("Christopher Nolan")
print("Bollywood")
print("Hollywood")
print("Marvel")
print("DC")
print("Action")
print("Horror")
print("Romance")
print("Comedy")
print("Drama")
print("Thriller")


# ============================================================
# USER INPUT
# ============================================================

user_input = input(
    "\nEnter movie, actor, director, "
    "industry or genre: "
)


# ============================================================
# NUMBER OF MOVIES
# ============================================================

count_input = input(
    "How many movies do you want? "
    "(default 15): "
)


if count_input.strip() == "":

    count = 15

else:

    try:

        count = int(
            count_input
        )

        if count <= 0:

            count = 15

    except ValueError:

        count = 15


# ============================================================
# RUN PROGRAM
# ============================================================

recommend(
    user_input,
    count
)