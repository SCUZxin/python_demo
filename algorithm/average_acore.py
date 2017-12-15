import random
import numpy as np

def make_score(num):
    score = np.random.randint(0, 100, num)
    return score

def less_average_count(score):
    average_score = score.mean()
    less_count = len([x for x in score if x < average_score])
    return less_count

if __name__ == "__main__":
    score = make_score(40)
    print(score)
    print(less_average_count(score))
    # 以下三种均为由高到低排序
    print(sorted(score, reverse=True))
    print('np.sort:', -(np.sort(-score)))
    print('np.sort:', np.sort(score)[::-1])
