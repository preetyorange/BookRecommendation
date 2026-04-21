import os
from typing import Tuple

import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BOOKS_PATH = os.path.join(DATA_DIR, "Books.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "Ratings.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "book_ratings.csv")


def _read_books() -> pd.DataFrame:
   
    books = pd.read_csv(BOOKS_PATH)

    # Standardize expected column names
    # Typical columns: 'ISBN', 'Book-Title', 'Book-Author', ...
    books = books.rename(
        columns={
            "Book-Title": "title",
            "ISBN": "ISBN",
        }
    )
    # Keep only columns we actually need
    books = books[["ISBN", "title"]].dropna(subset=["ISBN", "title"])
    return books


def _read_ratings() -> pd.DataFrame:
   
    ratings = pd.read_csv(RATINGS_PATH)

    ratings = ratings.rename(
        columns={
            "User-ID": "userId",
            "ISBN": "ISBN",
            "Book-Rating": "rating",
        }
    )

    required_cols = {"userId", "ISBN", "rating"}
    missing = required_cols - set(ratings.columns)
    if missing:
        raise ValueError(f"Ratings.csv is missing required columns: {missing}")

    return ratings


def filter_and_join(books: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
   
    # Remove implicit feedback (rating == 0)
    ratings = ratings[ratings["rating"] != 0]

    # Drop obvious nulls
    ratings = ratings.dropna(subset=["userId", "ISBN", "rating"])

    # Ensure numeric userId and rating
    ratings["userId"] = ratings["userId"].astype("int64", errors="ignore")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["rating"])

    # Filter users with >= 20 ratings
    user_counts = ratings.groupby("userId")["ISBN"].count()
    active_users = user_counts[user_counts >= 20].index
    ratings = ratings[ratings["userId"].isin(active_users)]

    # Filter books (ISBN) with >= 20 ratings
    book_counts = ratings.groupby("ISBN")["userId"].count()
    popular_isbns = book_counts[book_counts >= 20].index
    ratings = ratings[ratings["ISBN"].isin(popular_isbns)]

    if ratings.empty:
        raise ValueError(
            "After filtering for users/books with >= 20 ratings, no data remains. "
            "Try lowering the thresholds in prepare_data.py."
        )

    # Map ISBN -> numeric bookId
    unique_isbns = ratings["ISBN"].drop_duplicates().reset_index(drop=True)
    isbn_to_id = pd.Series(
        index=unique_isbns.values,
        data=(unique_isbns.index + 1).astype("int64"),
        name="bookId",
    )

    ratings = ratings.join(isbn_to_id, on="ISBN")

    # Join titles
    books_small = books.set_index("ISBN")
    ratings = ratings.join(books_small["title"], on="ISBN")

    # Final selection and ordering
    out = ratings[["userId", "bookId", "rating", "title"]].copy()
    out["userId"] = out["userId"].astype("int64")
    out["bookId"] = out["bookId"].astype("int64")
    out["rating"] = out["rating"].astype("float32")
    out = out.dropna(subset=["title"])

    return out


def main() -> None:
    print(f"Reading books from {BOOKS_PATH}")
    books = _read_books()
    print(f"Reading ratings from {RATINGS_PATH}")
    ratings = _read_ratings()

    print("Filtering and joining data...")
    prepared = filter_and_join(books, ratings)
    print(f"Final dataset size: {len(prepared)} rows")

    os.makedirs(DATA_DIR, exist_ok=True)
    prepared.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote prepared data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

