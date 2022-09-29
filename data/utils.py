from sklearn.model_selection import train_test_split
from collections import Counter


def train_val_test_split(paths, test_size=0.1, seed=42, verbose=True):
    train, test = train_test_split(paths, test_size=test_size, random_state=seed)
    train, val = train_test_split(train, test_size=test_size, random_state=seed)
    if verbose:
        n = len(paths)
        print(f"TRAIN SIZE: {len(train)} kitti files ({len(train) / n * 100}%)")
        print(f"VAL SIZE: {len(val)} kitti files ({len(val) / n * 100}%)")
        print(f"TEST SIZE: {len(test)} kitti files ({len(test) / n * 100}%)")
    return train, test, val


def most_frequent(arr):
    occurence_count = Counter(arr)
    return occurence_count.most_common(1)[0][0]