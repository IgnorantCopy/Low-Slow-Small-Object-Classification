import torch


def calc_rate(predictions):
    rate = 0.
    pred_strict = []
    for i in range(len(predictions)):
        pred = predictions[i]
        unique_values, counts = torch.unique(pred, return_counts=True)
        pred_label = unique_values[counts.argmax()]
        r = len(pred[pred == pred_label]) / len(pred)
        pred_strict.append(pred_label if r >= 0.9 else -1)
        rate += len(pred[pred == pred_label]) / len(pred)
    return rate / len(predictions), torch.tensor(pred_strict)
