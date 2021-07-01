from math import floor
import numpy as np
from scipy.misc import imresize

def Pads():
    pads = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    return pads

def crop_img(img, cy, cx, reg_s):
    
    assert reg_s % 2 != 0, "The region side must be an odd integer."
    
    pads = Pads()
    h, w, _ = img.shape
    context = (reg_s-1)/2  # The amount added in each direction
    xcrop_min = int(floor(cx) - context)
    xcrop_max = int(floor(cx) + context)
    ycrop_min = int(floor(cy) - context)
    ycrop_max = int(floor(cy) + context)
    # Check if any of the corners exceeds the boundaries of the image.
    if xcrop_min < 0:
        pads['left'] = -(xcrop_min)
        xcrop_min = 0
    if ycrop_min < 0:
        pads['up'] = -(ycrop_min)
        ycrop_min = 0
    if xcrop_max >= w:
        pads['right'] = xcrop_max - w + 1
        xcrop_max = w - 1
    if ycrop_max >= h:
        pads['down'] = ycrop_max - h + 1
        ycrop_max = h - 1
    cropped_img = img[ycrop_min:(ycrop_max+1), xcrop_min:(xcrop_max+1)]

    return cropped_img, pads


def resize_and_pad(cropped_img, out_sz, pads, reg_s=None, use_avg=True):
    cr_h, cr_w, _ = cropped_img.shape
    if reg_s:
        assert ((cr_h+pads['up']+pads['down'] == reg_s) and
                (cr_w+pads['left']+pads['right'] == reg_s)), (
            'The informed crop dimensions and pad amounts are not consistent '
            'with the informed region side. Cropped img shape: {}, Pads: {}, '
            'Region size: {}.'
            .format(cropped_img.shape, pads, reg_s))
    rz_ratio = out_sz/(cr_h + pads['up'] + pads['down'])
    rz_cr_h = round(rz_ratio*cr_h)
    rz_cr_w = round(rz_ratio*cr_w)
    
    pads['up'] = round(rz_ratio*pads['up'])
    pads['down'] = out_sz - (rz_cr_h + pads['up'])
    pads['left'] = round(rz_ratio*pads['left'])
    pads['right'] = out_sz - (rz_cr_w + pads['left'])
    # Notice that this resized crop is not necessarily a square.
    rz_crop = imresize(cropped_img, (rz_cr_h, rz_cr_w), interp='bilinear')
    # Differently from the paper here we are using the mean of all channels
    # not on each channel. It might be a problem, but the solution might add a lot
    # of overhead
    if use_avg:
        const = np.mean(cropped_img)
    else:
        const = 0
    # Pads only if necessary, i.e., checks if all pad amounts are zero.
    if not all(p == 0 for p in pads.values()):
        out_img = np.pad(rz_crop, ((pads['up'], pads['down']),
                                   (pads['left'], pads['right']),
                                   (0, 0)),
                         mode='constant',
                         constant_values=const)
    else:
        out_img = rz_crop
    return out_img
