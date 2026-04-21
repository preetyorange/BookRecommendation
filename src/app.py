import os

from flask import Flask, render_template, request
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECS_PATH = os.path.join(BASE_DIR, "data", "user_recommendations.csv")
RATINGS_PATH = os.path.join(BASE_DIR, "data", "book_ratings.csv")


def create_app() -> Flask:
    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

    if not os.path.exists(RECS_PATH):
        raise FileNotFoundError(
            f"Recommendations file not found at {RECS_PATH}. "
            f"Please run the training script first: python src/train_als.py"
        )

    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(
            f"Ratings file not found at {RATINGS_PATH}. "
            f"Please run the data prep first: python src/prepare_data.py"
        )

    # Load prepared ratings for popularity + counts + search (pure pandas)
    ratings_df = pd.read_csv(
        RATINGS_PATH,
        usecols=["userId", "bookId", "rating", "title"],
        dtype={
            "userId": "int64",
            "bookId": "int64",
            "rating": "float32",
            "title": "string",
        },
    ).rename(columns={"userId": "user_id", "bookId": "book_id"})

    book_stats = (
        ratings_df.groupby(["book_id", "title"], as_index=False)
        .agg(rated_by_users=("user_id", "nunique"), avg_rating=("rating", "mean"))
    )

    # Use explicit dtypes to keep memory usage reasonable on larger datasets
    recs_df = pd.read_csv(
        RECS_PATH,
        dtype={
            "user_id": "int64",
            "book_id": "int64",
            "score": "float32",
        },
    )
    recs_df = recs_df.merge(book_stats[["book_id", "rated_by_users"]], on="book_id", how="left")
    recs_df["rated_by_users"] = recs_df["rated_by_users"].fillna(0).astype("int64")
    # Optional index to speed up lookups by user_id
    recs_df.set_index("user_id", inplace=True)

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            results=None,
            user_id=None,
            error=None,
            search_results=None,
            query=None,
        )

    @app.route("/recommend", methods=["POST"])
    def recommend():
        user_id_str = request.form.get("user_id", "").strip()
        if not user_id_str.isdigit():
            return render_template(
                "index.html",
                results=None,
                user_id=user_id_str,
                error="Please enter a valid numeric user ID.",
                search_results=None,
                query=None,
            )

        user_id = int(user_id_str)
        try:
            user_recs = recs_df.loc[user_id]
        except KeyError:
            user_recs = pd.DataFrame(columns=["book_id", "score", "title", "rated_by_users"])

        if not isinstance(user_recs, pd.DataFrame):
            user_recs = user_recs.to_frame().T

        user_recs = (
            user_recs.drop_duplicates(subset=["book_id"])
            .sort_values(by="score", ascending=False)
        )

        if user_recs.empty:
            return render_template(
                "index.html",
                results=[],
                user_id=user_id,
                error="No recommendations found for this user. Try another user ID.",
                search_results=None,
                query=None,
            )

        records = user_recs.to_dict(orient="records")

        return render_template(
            "index.html",
            results=records,
            user_id=user_id,
            error=None,
            search_results=None,
            query=None,
        )

    @app.route("/search", methods=["POST"])
    def search():
        query = (request.form.get("query") or "").strip()
        if not query:
            return render_template(
                "index.html",
                results=None,
                user_id=None,
                error="Please enter a story title to search.",
                search_results=None,
                query=query,
            )

        # Find matching titles
        matches = ratings_df[ratings_df["title"].str.contains(query, case=False, na=False)]
        if matches.empty:
            return render_template(
                "index.html",
                results=None,
                user_id=None,
                error=f'No titles found matching "{query}".',
                search_results=None,
                query=query,
            )

        # Pick the most-rated matching title as the "target"
        match_counts = matches.groupby(["book_id", "title"], as_index=False).agg(
            rating_count=("user_id", "nunique"),
            avg_rating=("rating", "mean"),
        )
        match_counts = match_counts.sort_values(["rating_count", "avg_rating"], ascending=False)
        target = match_counts.iloc[0].to_dict()
        target_book_id = int(target["book_id"])

        # Users who rated this target highly
        high_users = matches[(matches["book_id"] == target_book_id) & (matches["rating"] >= 8.0)][
            ["user_id", "rating"]
        ].sort_values("rating", ascending=False)

        # Similar stories: other stories those users liked (rating>=8)
        if high_users.empty:
            similar = pd.DataFrame(columns=["title", "book_id", "avg_rating", "rated_by_users"])
        else:
            user_ids = high_users["user_id"].unique()
            similar_raw = ratings_df[
                (ratings_df["user_id"].isin(user_ids))
                & (ratings_df["rating"] >= 8.0)
                & (ratings_df["book_id"] != target_book_id)
            ]
            similar = (
                similar_raw.groupby(["book_id", "title"], as_index=False)
                .agg(avg_rating=("rating", "mean"), rated_by_users=("user_id", "nunique"))
                .sort_values(["rated_by_users", "avg_rating"], ascending=False)
                .head(10)
            )

        search_results = {
            "query": query,
            "target": target,
            "top_matches": match_counts.head(10).to_dict(orient="records"),
            "high_users": high_users.head(10).to_dict(orient="records"),
            "similar": similar.to_dict(orient="records"),
        }

        return render_template(
            "index.html",
            results=None,
            user_id=None,
            error=None,
            search_results=search_results,
            query=query,
        )

    @app.route("/popular", methods=["GET"])
    def popular():
        most_rated = (
            book_stats.sort_values("rated_by_users", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

        highest_avg = (
            book_stats[book_stats["rated_by_users"] >= 20]
            .sort_values("avg_rating", ascending=False)
            .head(10)
            .to_dict(orient="records")
        )

        return render_template(
            "popular.html",
            most_rated=most_rated,
            highest_avg=highest_avg,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
