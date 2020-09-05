import torch
def CRPSLoss(y_pred, y):
    return torch.mean((y_pred.cumsum(-1) - y.cumsum(-1))**2)
