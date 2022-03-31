from typing import List

import numpy as np
import pandas as pd
import torch.utils.data as data
from scipy.sparse import dok_matrix

from .helper import load_config

cfg = load_config("./config/config.yaml")


class NCFData(data.Dataset):
    r"""Dataset class for Neural Collaborative Filtering.

    Parameters
    ----------
    data : List[List[int]]
        A two-dimensional list is required.
        Every element in the list has a form that `[user_id, item_id]`
    num_items : int
        The number of items
    train_mat : dok_matrix, optional
        A sparse matrix which contains training data, by default None
    num_negative_samples : int, optional
        The number of negative samples, by default 0
        It is only used when training scenario.
    is_training : bool, optional
        By default None
    """

    def __init__(
        self,
        data: List[List[int]],
        num_items: int,
        train_mat: dok_matrix = None,
        num_negative_samples: int = 0,
        is_training: bool = None,
    ):
        super(NCFData, self).__init__()

        self.pos_data = data
        self.num_items = num_items
        self.train_mat = train_mat
        self.num_negative_samples = num_negative_samples
        self.is_training = is_training
        self.labels = [0] * len(data)

    def __len__(self):
        return (self.num_negative_samples + 1) * len(self.labels)

    def __getitem__(self, idx):
        data = self.data_fill if self.is_training else self.pos_data
        labels = self.labels_fill if self.is_training else self.labels

        user = data[idx][0]
        item = data[idx][1]
        label = labels[idx]
        return user, item, label

    def negative_sampling(self):
        assert (
            self.is_training
        ), "No need to negative sampling when testing scenario."

        self.neg_data = []
        neg_data_by_user = {}
        for x in self.pos_data:
            user_id = x[0]
            if user_id not in neg_data_by_user.keys():
                neg_data_by_user[user_id] = np.setdiff1d(
                    np.arange(self.num_items),
                    self.train_mat[user_id].nonzero()[1],
                )
            negatives = neg_data_by_user.get(user_id)
            self.neg_data.extend(
                [user_id, item_id]
                for item_id in np.random.choice(
                    negatives, self.num_negative_samples
                )
            )
        pos_labels = [1] * len(self.pos_data)
        neg_labels = [0] * len(self.neg_data)

        self.data_fill = self.pos_data + self.neg_data
        self.labels_fill = pos_labels + neg_labels


def get_dataset():
    r"""Load data to construct dataset for Torch"""
    train_data = pd.read_csv(
        cfg["data"]["train_rating"],
        sep="\t",
        header=None,
        names=["user", "item"],
        usecols=[0, 1],
        dtype={0: np.int32, 1: np.int32},
    )

    num_users = np.max(train_data["user"]) + 1
    num_items = np.max(train_data["item"]) + 1

    train_data = train_data.values.tolist()

    # convert train data into the dok format
    train_mat = dok_matrix((num_users, num_items), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # transform test data into the same format of train data
    test_data = []
    with open(cfg["data"]["test_negative"], "r") as fd:
        lines = fd.readlines()
        for line in lines:
            arr = line.split("\t")
            user_id = eval(arr[0])[0]
            test_data.append([user_id, eval(arr[0])[1]])
            test_data.extend([user_id, int(i)] for i in arr[1:])
    return train_data, test_data, num_users, num_items, train_mat
