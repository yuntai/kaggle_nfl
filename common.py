import torch
def CRPSLoss(y_pred, y):
    return torch.sum((y_pred.cumsum(-1) - y.cumsum(-1))**2) / (199 * y_pred.shape[0])

YARDS_CLIP = [-15, 50]
