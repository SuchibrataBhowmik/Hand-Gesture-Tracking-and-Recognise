import numpy as np
from sklearn.metrics import roc_auc_score


def center_error(output, label, upscale_factor=4):
    b = output.shape[0]
    s = output.shape[-1]
    out_flat = output.reshape(b, -1)
    max_idx = np.argmax(out_flat, axis=1)
    estim_center = np.stack([max_idx//s, max_idx % s], axis=1)
    dist = np.linalg.norm(estim_center - s//2, axis=1)
    c_error = dist.mean()
    c_error = c_error * upscale_factor
    return c_error

def AUC(output, label):
    b = output.shape[0]
    output = output.reshape(b, -1)
    mask = label[:, :, :, 1].reshape(b, -1)
    label = label[:, :, :, 0].reshape(b, -1)
    total_auc = 0
    for i in range(b):
        total_auc += roc_auc_score(label[i], output[i], sample_weight=mask[i])
    return total_auc/b


METRICS = {'AUC': {'fcn': AUC, 'kwargs': {}},
           'center_error': {'fcn': center_error, 'kwargs': {'upscale_factor': 4}}
          }
