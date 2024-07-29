import torch
from torch.utils.data import Dataset


class MovieRatingDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.users = self.data["userId"].unique()
        self.movies = self.data["movieId"].unique()
        self.user_to_index = {user: idx for idx, user in enumerate(self.users)}
        self.movie_to_index = {movie: idx for idx, movie in enumerate(self.movies)}
        self.users_len = len(self.users)
        self.movies_len = len(self.movies)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_idx = self.user_to_index[row["userId"]]
        movie_idx = self.movie_to_index[row["movieId"]]
        rating = row["rating"]
        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(movie_idx, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float32),
        )

    def get_user_movies(self, user_id):
        if user_id not in self.user_to_index:
            raise ValueError(f"User ID {user_id} not found in dataset.")

        user_movies = self.data[self.data["userId"] == user_id]["movieId"].values
        return user_movies

    def user_len(self):
        return self.users_len

    def movie_len(self):
        return self.movies_len

    def movie_ids(self):
        return self.movies

    def get_user_ids(self):
        return self.users
