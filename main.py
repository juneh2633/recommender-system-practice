from scripts.train import train
from utils.recommend_movie import recommend


def main():

    while 1:
        print("1. train ")
        print("2. load ")
        a = int(input("input 1 or 2 \n"))
        if a == 1:
            train()
            break
        elif a == 2:
            user_idx = int(input("input user idx \n"))
            recommend(user_idx)
            break


if __name__ == "__main__":
    main()
