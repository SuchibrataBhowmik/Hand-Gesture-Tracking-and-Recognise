from argparse import ArgumentParser
import os 
from os.path import join,isfile, exists
import xml.etree.ElementTree as ET
from math import floor
import numpy as np
import cv2
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


def make_ref(ref_path):
    annopath = ref_path.replace('jpeg', 'xml')
    ref_frame = imread(ref_path)

    annotation, width, height, valid_frame = get_annotations(annopath)
    xywh = [ annotation['xmin']/width, annotation['ymin']/height, (annotation['xmax']-annotation['xmin'])/width, (annotation['ymax']-annotation['ymin'])/height ]
    vid_dims= (height,width)
    resize_dims = None
    ref_img = extract_ref(ref_frame, xywh, vid_dims)
    return ref_img
def get_annotations(annot_path):
    if isfile(annot_path):
        tree = ET.parse(annot_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if root.find('object') is None:
            annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
            valid_frame = False
        else:
            obj = root.find('object')
            if obj.find('bndbox'):
                bbox = obj.find('bndbox')
                xmax = int(float(bbox.find('xmax').text))
                xmin = int(float(bbox.find('xmin').text))
                ymax = int(float(bbox.find('ymax').text))
                ymin = int(float(bbox.find('ymin').text))
                annotation = {'xmax': xmax, 'xmin': xmin, 'ymax': ymax, 'ymin': ymin}
                valid_frame = True
            else:
                annotation = {'xmax': None, 'xmin': None, 'ymax': None, 'ymin': None}
                valid_frame = False
    else:
        raise FileNotFoundError("The file {} could not be found" .format(annot_path))
    return annotation, width, height, valid_frame
def extract_ref(ref_frame,xywh,vid_dims):
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

def search(ref_emb, srch_emb, net):
        score_map = net.match_corr(ref_emb, srch_emb)

        dimx = score_map.shape[-1]
        dimy = score_map.shape[-2]
        score_map = score_map.view(-1, dimy, dimx)
        score_map = sigmoid(score_map)
        score_map = score_map.unsqueeze(0)
        # We upscale 4 times, because the total stride of the network is 4
        score_map = F.interpolate(score_map, scale_factor=4, mode='bilinear', align_corners=False)
        score_map = score_map.cpu()
        score_map = torch_var_to_numpy(score_map)
        peak = np.unravel_index(score_map.argmax(), score_map.shape)
        
        return peak,score_map.max()

def argparser():
    parser = ArgumentParser(description="Evaluation for hand gesture")
    parser.add_argument("-m","--model", type=str, default='experiment/best.pth.tar',  help="Enter the model directory")
    parser.add_argument("-d","--testdata", type=str, default="classification/vid1",  help="Enter the test dataset directory")
    parser.add_argument("-s","--save", default=False,  help="Do you save the output ? If yes type True")
    parser.add_argument("-o","--output_file", type=str, default='classification.mp4',  help="Enter the output file directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()

    model_path = args.model
    testset = args.testdata
    if not exists(model_path) and exists(testset):
        raise Exception("Directories are not exists")

    gesture1 = 'classification/open.jpeg'
    gesture2 = 'classification/close.jpeg'
    gesture_img1 = make_ref(gesture1)
    gesture_img2 = make_ref(gesture2)
    #imwrite('classification/open_ges.jpeg',gesture_img1)
    #imwrite('classification/close_ges.jpeg',gesture_img2)
    
    model = get_model(model_path)
    emb_gesture1 = get_emb(gesture_img1, model)
    emb_gesture2 = get_emb(gesture_img2, model)
    
    if args.save:
        width, height = 960, 540
        size = (width, height)
        fps = 15
        output = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'),fps, size)

    for frame in sorted(os.listdir(testset)):
        frm_path = join(testset,frame)
        print(frm_path)
        srch_img = imread(frm_path)
        org_img = srch_img.copy()

        srch_img = srch_img/255
        offset = (((gesture_img1.shape[0] + 1)//4)*4 - 1)//2      # gesture_img1.shape[0] = 127
        srch_img_mean = srch_img.mean()
        srch_img_padded = np.pad(srch_img, ((offset, offset), (offset, offset), (0, 0)), mode='constant', constant_values=srch_img_mean)
        emb_srch_img = get_emb(srch_img_padded, model)

        ges1_peak,ges1_accu = search(emb_gesture1, emb_srch_img, model)
        ges2_peak,ges2_accu = search(emb_gesture2, emb_srch_img, model)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
        
        if ges1_accu >= 0.90 and ges1_accu>ges2_accu :
            cv2.circle(org_img,(ges1_peak[1],ges1_peak[0]),100,(0,255,0),2)
            cv2.putText(org_img, 'open', (70,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
        elif ges2_accu >= 0.90 :
            cv2.circle(org_img,(ges2_peak[1],ges2_peak[0]),70,(0,255,0),2)
            cv2.putText(org_img, 'close', (70,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            
        if args.save : output.write(org_img)
        cv2.imshow("result",org_img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:break
    if args.save : output.release()
    cv2.destroyAllWindows()


