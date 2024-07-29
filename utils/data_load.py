import pandas as pd

from dataset.MovieRatingDatasetClass import MovieRatingDataset


def dataset_split(file_path, train_ratio=0.01, val_ratio=0.005, test_ratio=0.005):
    print("load data")
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    print(df)

    print("sort data")
    df = df.sort_values(by="timestamp")

    num_rows = len(df)
    print(num_rows)
    train_size = int(num_rows * train_ratio)
    val_size = int(num_rows * val_ratio)
    # test_size = num_rows - train_size - val_size
    test_size = int(num_rows * test_ratio)
    print("train size = ", train_size)
    print("val size = ", val_size)
    print("test size = ", test_size)
    train_df = df[:train_size]
    val_df = df[train_size : train_size + val_size]
    # test_df = df[train_size + val_size :]
    # index에러 발생
    test_df = df[train_size + val_size : train_size + val_size + test_size]
    train_dataset = MovieRatingDataset(train_df)
    val_dataset = MovieRatingDataset(val_df)
    test_dataset = MovieRatingDataset(test_df)
    print("split success")
    return train_dataset, val_dataset, test_dataset


def dataset_by_user_idx(file_path, user_idx):
    df = pd.read_csv(file_path)

    user_data = df[df["userId"] == user_idx]

    return MovieRatingDataset(user_data)


def load_movie_titles():
    movies_df = pd.read_csv("data/movie.csv")
    movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    return movie_id_to_title
