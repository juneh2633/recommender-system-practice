import torch


def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for user_indices, movie_indices, ratings in test_loader:
            outputs = model(user_indices, movie_indices)
            loss = criterion(outputs, ratings)
            test_loss += loss.item()
    return test_loss / len(test_loader)
