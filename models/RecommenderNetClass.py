import torch
import torch.nn as nn


class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.movie_embedding = nn.Embedding(num_movies, embedding_size)

        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, user_indices, movie_indices):

        user_embedding = self.user_embedding(user_indices)
        movie_embedding = self.movie_embedding(movie_indices)

        user_bias = self.user_bias(user_indices).squeeze()
        movie_bias = self.movie_bias(movie_indices).squeeze()

        dot_product = (user_embedding * movie_embedding).sum(1)

        rating = dot_product + user_bias + movie_bias

        rating = torch.clamp(rating, 0.0, 5.0)

        return rating
