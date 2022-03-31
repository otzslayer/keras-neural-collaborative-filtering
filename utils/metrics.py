import numpy as np
import torch


def hit(actual, preds):
    return 1 if actual in preds else 0


def ndcg(actual, preds):
    if actual in preds:
        index = preds.index(actual)
        return np.reciprocal(np.log2(index + 2))
    else:
        return 0


def get_metrics(model, test_loader, top_k):
    hit_list, ndcg_list = [], []

    for user, item, _ in test_loader:
        user = user.cuda()
        item = item.cuda()

        preds = model(user, item)
        _, indices = torch.topk(preds, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        pos_item = item[0].item()
        hit_list.append(hit(pos_item, recommends))
        ndcg_list.append(ndcg(pos_item, recommends))

    return np.mean(hit_list), np.mean(ndcg_list)
