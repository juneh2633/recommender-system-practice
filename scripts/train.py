import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.RecommenderNetClass import RecommenderNet
from scripts.evaluate_model import evaluate_model
from utils.data_load import dataset_split


def train():

    train_dataset, val_dataset, test_dataset = dataset_split("data/rating.csv")

    # DataLoader 정의
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print("data load success")
    num_users = train_dataset.user_len()
    num_movies = train_dataset.movie_len()
    embedding_size = 50  # Example embedding size

    model = RecommenderNet(num_users, num_movies, embedding_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for i, (user_indices, movie_indices, ratings) in enumerate(train_loader):
            if i % 10 == 0:
                print(f"Processing batch {i} in epoch {epoch + 1}")

            optimizer.zero_grad()
            outputs = model(user_indices, movie_indices)

            if i % 10 == 0:
                print(f"Outputs: {outputs}, Ratings: {ratings}")

            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 10 == 0:
                print(f"Loss for batch {i} in epoch {epoch + 1}: {loss.item()}")

        print(f"Epoch {epoch + 1}, Train Loss: {epoch_loss/len(train_loader)}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_indices, movie_indices, ratings in val_loader:
                outputs = model(user_indices, movie_indices)
                loss = criterion(outputs, ratings)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss/len(val_loader)}")

        print("Training complete")

        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, "recommend_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        test_loss = evaluate_model(model, test_loader, criterion)
        print(f"Test Loss: {test_loss}")
