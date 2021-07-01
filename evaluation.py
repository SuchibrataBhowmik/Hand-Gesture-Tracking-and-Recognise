from argparse import ArgumentParser
import os 
from os.path import join, exists
from math import floor
import numpy as np
import cv2
from imageio import imread
from scipy.misc import imresize
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import torch
import torch.nn.functional as F
from torch import sigmoid

import library.models as mdl
from library.train_utils import get_annotations
from library.tensor_conv import numpy_to_torch_var, torch_var_to_numpy

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")
print('Device = ',device)


def make_ref(frmpath,frame):
    ref = join(frmpath,frame)
    ref_frame = imread(ref)

    refanno =  join('/'.join(frmpath.split('/')[:-1]),'annotations')
    annotation, width, height, valid_frame = get_annotations(refanno, frame)
    
    xywh = [ annotation['xmin']/width, annotation['ymin']/height, (annotation['xmax']-annotation['xmin'])/width, (annotation['ymax']-annotation['ymin'])/height ]
    vid_dims= (height,width)
    resize_dims = None
    ref_img = extract_ref(ref_frame, xywh, vid_dims)
    return ref_img
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
        
        return peak,round(score_map.max(),3)


def argparser():
    parser = ArgumentParser(description="Evaluation for hand gesture")

    parser.add_argument("-m","--model", type=str, default='experiment/best.pth.tar',  help="Enter the model directory")
    parser.add_argument("-d","--data", type=str, default="hand_dataset/train/0001/data",  help="Enter the data directory")
    parser.add_argument("-a","--annotations", type=str, default="hand_dataset/train/0001/annotations",  help="Enter the data directory")
    parser.add_argument("-of","--object_frame", type=str, default='000000.jpeg',  help="Enter the first object frame")

    parser.add_argument("-s","--save", default=False,  help="Do you save the output ? If yes type True")
    parser.add_argument("-o","--output_file", type=str, default='evaluation.mp4',  help="Enter the output file directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()

    model_path = args.model
    data = args.data
    annotations = args.annotations
    object_frame = args.object_frame
    if not exists(model_path) and exists(data) and exists(annotations):
        raise Exception("Directories are not exists")

    if args.save:
        width, height = 1400, 450
        size = (width, height)
        fps = 15
        output = cv2.VideoWriter(args.output_file, cv2.VideoWriter_fourcc(*'MP4V'),fps, size)

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.margins(0)

    model = get_model(model_path)
    ref_obj = make_ref(data,object_frame)
    emb_obj = get_emb(ref_obj, model)
    
    frm_no, cen_err = [], []
    for i, frame in enumerate(sorted(os.listdir(data))):
        #if i%2 == 0: continue
        annotation, width, height, valid_frame = get_annotations(annotations,frame)
        frm_path = join(data,frame)
        print(frm_path)
        srch_img = imread(frm_path)
        org_img = srch_img.copy()

        srch_img = srch_img/255
        offset = (((ref_obj.shape[0] + 1)//4)*4 - 1)//2      # obj.shape[0] = 127
        srch_img_mean = srch_img.mean()
        srch_img_padded = np.pad(srch_img, ((offset, offset), (offset, offset), (0, 0)), mode='constant', constant_values=srch_img_mean)
        
        org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
        
        emb_srch_img = get_emb(srch_img_padded, model)
        peak,score = search(emb_obj, emb_srch_img, model)
        cv2.circle(org_img,(peak[1],peak[0]),2,(0,0,255),2)
        
        center_error = None
        if valid_frame:
            org_center = (int((annotation['xmin']+annotation['xmax'])/2), int((annotation['ymin']+annotation['ymax'])/2))
            cv2.circle(org_img,(org_center[0],org_center[1]),4,(0,255,0),4)
            center_error = round( np.linalg.norm([(org_center[0]-peak[1]), (org_center[1]-peak[0])]), 3)
            if center_error < 40:
                cv2.circle(org_img,(peak[1],peak[0]),80,(0,0,255),2)

        cv2.putText(org_img, 'frm_no : '+str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(org_img, 'center_err : '+str(center_error), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(org_img, 'score : '+str(score), (50,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        frm_no.append(i)
        cen_err.append(center_error)
        ax.plot(frm_no,cen_err,'g')
        frms = []
        for _ in range(i):frms.append(40)
        ax.plot(np.arange(i),frms, 'b')
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = cv2.resize(image_from_plot,(600,450))

        org_img = cv2.resize(org_img,(800,450))
        
        concat_img = cv2.hconcat([org_img, image_from_plot])
        if args.save : output.write(concat_img) 
        cv2.imshow('result',concat_img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:break
    if args.save : output.release()
    cv2.destroyAllWindows()
    
            


