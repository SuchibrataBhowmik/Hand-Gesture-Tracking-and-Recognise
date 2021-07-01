from argparse import ArgumentParser
from os.path import join, exists
from math import floor
import numpy as np
from matplotlib import cm
import cv2
import os 
from imageio import imread
from scipy.misc import imresize

import torch
import torch.nn.functional as F
from torch import sigmoid

import library.models as mdl
from library.tensor_conv import numpy_to_torch_var, torch_var_to_numpy


device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
print('Device = ',device)

resize_dims = None

def get_model(model_path):
    net = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), stride=4)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['state_dict'])
    net = net.to(device)
    net.eval()
    return net
@torch.no_grad()
def get_emb(img, net):
    img_tensor = numpy_to_torch_var(img, device)
    emb_img = net.get_embedding(img_tensor)
    return emb_img 

def make_ref(frame):
    ref_frame = imread(frame)
    ref = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR)
    bb = cv2.selectROI('Frame',ref)
    cv2.destroyAllWindows()
    
    height,width = ref_frame.shape[0],ref_frame.shape[1]
    xywh = [ bb[0]/width, bb[1]/height, bb[2]/width, bb[3]/height ]
    vid_dims= (height,width)
    
    ref_img = extract_ref(ref_frame, xywh, vid_dims)
    return ref_img
def extract_ref(ref_frame,xywh,vid_dims):
    global resize_dims
    bbox = denorm_bbox(xywh, vid_dims)
    ctx_size = max(bbox[2], bbox[3])
    if ctx_size != 127:
        new_H = int(vid_dims[0]*127/ctx_size)
        new_W = int(vid_dims[1]*127/ctx_size)
        resize_dims = (new_H, new_W)
        ref_frame = imresize(ref_frame, resize_dims, interp='bilinear')
        bbox = denorm_bbox(xywh, resize_dims)
        ctx_size = 127
    ref_frame = ref_frame/255
    ref_center = (int((bbox[1] + bbox[3]/2)), int((bbox[0] + bbox[2]/2)))
    ref_img = crop(ref_frame, ref_center, ctx_size)
    return ref_img
def denorm_bbox(bbox_norm, img_dims):
    bbox = bbox_norm[:]
    bbox[0] = int(bbox[0]*img_dims[1])
    bbox[1] = int(bbox[1]*img_dims[0])
    bbox[2] = int(floor(bbox[2]*img_dims[1]))
    bbox[3] = int(floor(bbox[3]*img_dims[0]))
    return tuple(bbox)
def crop(ref_frame, center, ctx_size):
    H, W, _ = ref_frame.shape
    y_min = max(0, center[0]-ctx_size//2)
    y_max = min(H-1, center[0] + ctx_size//2)
    x_min = max(0, center[1]-ctx_size//2)
    x_max = min(W-1, center[1] + ctx_size//2)
    offset_top = max(0, ctx_size//2 - center[0])
    offset_bot = max(0, center[0] + ctx_size//2 - H + 1)
    offset_left = max(0, ctx_size//2 - center[1])
    offset_right = max(0, center[1] + ctx_size//2 - W + 1)
    img_mean = ref_frame.mean()
    ref_img = np.ones([ctx_size, ctx_size, 3])*img_mean
    ref_img[offset_top:(ctx_size-offset_bot), offset_left:(ctx_size-offset_right)] = ( ref_frame[y_min:(y_max+1), x_min:(x_max+1)] )      
    return ref_img

def search(emb_obj, frm_path, model):
    global resize_dims
    srch_img = imread(frm_path)
    if resize_dims is not None:
        srch_img = imresize(srch_img, resize_dims, interp='bilinear')

    score_map = make_score_map(emb_obj, srch_img, model)    
    peak = np.unravel_index(score_map.argmax(), score_map.shape)
            
    score_img = cm.inferno(score_map)[:, :, 0:3]        # Convert 1 chanel to 3 chanel
    score_img = score_img[0:srch_img.shape[0], 0:srch_img.shape[1], :]      # reduce size as search imag
    score_img = score_img+ srch_img/255     # Add two image
    score_img = (score_img * 255).astype(np.uint8)      # Additional image exceed between 0,1 so, convert to integre 
    res_img = cv2.cvtColor(srch_img, cv2.COLOR_RGB2BGR)
    cv2.circle(res_img,(peak[1],peak[0]),2,(0,0,255),2)
    cv2.rectangle(res_img,(peak[1]-65,peak[0]-65),(peak[1]+65,peak[0]+65),255,2)

    return res_img
def make_score_map(emb_obj, img, model):
    img = img/255
    offset = (((127+ 1)//4)*4 - 1)//2     # ref.shape[0] = 127
    img_mean = img.mean()
    img_padded = np.pad(img, ((offset, offset), (offset, offset), (0, 0)), mode='constant', constant_values=img_mean)
    
    srch_emb = get_emb(img_padded, model)
    score_map = model.match_corr(emb_obj, srch_emb)

    dimx = score_map.shape[-1]
    dimy = score_map.shape[-2]
    score_map = score_map.view(-1, dimy, dimx)
    score_map = sigmoid(score_map)
    score_map = score_map.unsqueeze(0)
    # We upscale 4 times, because the total stride of the network is 4
    score_map = F.interpolate(score_map, scale_factor=4, mode='bilinear', align_corners=False)
    score_map = score_map.cpu()
    score_map = torch_var_to_numpy(score_map)
    return score_map

def argparser():
    parser = ArgumentParser(description="Evaluation for hand gesture")

    parser.add_argument("-m","--model", type=str, default='experiment/best.pth.tar',  help="Enter the model directory")
    parser.add_argument("-d","--testdata", type=str, default="hand_dataset/test/0001",  help="Enter the test dataset directory")
    parser.add_argument("-of","--object_frame", type=str, default='000000.jpeg',  help="Enter the first object frame")

    parser.add_argument("-s","--save", default=False,  help="Do you save the output ? If yes type True")
    parser.add_argument("-o","--output_file", type=str, default='track.mp4',  help="Enter the output file directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()

    model_path = args.model
    testset = args.testdata
    object_frame = args.object_frame
    object_frame = join(testset, object_frame)

    if not exists(model_path) and exists(testset) and exists(object_frame):
        raise Exception("Directories are not exists")

    model = get_model(model_path)
    ref_obj = make_ref(object_frame)
    emb_obj = get_emb(ref_obj, model)

    if args.save:
        width, height = 960, 540
        size = (width, height)
        fps = 15
        output = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'),fps, size)

    for i, frame in enumerate(sorted(os.listdir(testset))):
        #if i%2 == 0: continue
        frm_path = join(testset,frame)
        print(frm_path)
        tracked_img = search(emb_obj,frm_path,model)
        tracked_img = cv2.resize(tracked_img,(960,540))

        if args.save : output.write(tracked_img)
        cv2.imshow('frame',tracked_img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:break
    if args.save : output.release()
    cv2.destroyAllWindows()
   
    


