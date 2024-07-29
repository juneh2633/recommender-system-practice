import torch

from models.RecommenderNetClass import RecommenderNet
from utils.data_load import MovieRatingDataset, dataset_split, load_movie_titles


def load_model(model_path, num_users, num_movies, embedding_size):
    model = RecommenderNet(num_users, num_movies, embedding_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def prepare_data(rating_csv_path):
    dataset = MovieRatingDataset(rating_csv_path)
    num_users = dataset.user_len()
    num_movies = dataset.movie_len()
    return dataset, num_users, num_movies


def list_up_movie_data(user_id, model, dataset, movie_id_to_title, limit=10):
    all_movie_ids = dataset.movie_ids()
    user_movies = dataset.get_user_movies(user_id)
    movie_scores = []
    user_index = dataset.user_to_index[user_id]

    for movie_id in all_movie_ids:
        if movie_id not in user_movies:  # 사용자가 이미 본 영화는 제외
            with torch.no_grad():
                movie_index = dataset.movie_to_index[movie_id]
                score = model(
                    torch.tensor([user_index]), torch.tensor([movie_index])
                ).item()
                movie_scores.append((movie_id, score))

    movie_scores.sort(key=lambda x: x[1], reverse=True)
    recommended_movies = movie_scores[:limit]

    recommendations_with_titles = [
        (movie_id_to_title.get(movie_id, "Unknown Title"), score)
        for movie_id, score in recommended_movies
    ]

    return recommendations_with_titles


def recommend(user_id):
    model_path = "models/recommend_model.pth"
    rating_csv_path = "data/rating.csv"

    dataset, _, _ = dataset_split(rating_csv_path)

    movie_id_to_title = load_movie_titles()

    # 모델 로드
    num_users = dataset.user_len()
    num_movies = dataset.movie_len()
    print("All user IDs in the dataset:")
    user_list = dataset.get_user_ids()
    sorted_user_list = sorted(user_list, reverse=True)
    for id in sorted_user_list:
        print(id)

    model = load_model(model_path, num_users, num_movies, embedding_size=50)

    recommendations = list_up_movie_data(user_id, model, dataset, movie_id_to_title)

    print(f"Recommended movies for user {user_id}:")
    for title, score in recommendations:
        print(f"Movie Title: {title}, Predicted Score: {score}")
