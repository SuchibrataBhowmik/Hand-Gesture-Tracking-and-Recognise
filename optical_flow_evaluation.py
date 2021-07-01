from argparse import ArgumentParser
import numpy as np
import cv2
import os
from os.path import join, exists
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from library.train_utils import get_annotations


def argparser():
    parser = ArgumentParser(description="Evaluation for hand gesture")

    parser.add_argument("-d","--data", type=str, default="hand_dataset/train/0001/data",  help="Enter the data directory")
    parser.add_argument("-a","--annotations", type=str, default="hand_dataset/train/0001/annotations",  help="Enter the data directory")
    parser.add_argument("-of","--object_frame", type=str, default='000000.jpeg',  help="Enter the first object frame")

    parser.add_argument("-s","--save", default=False,  help="Do you save the output ? If yes type True")
    parser.add_argument("-o","--output_file", type=str, default='optical_flow_evaluation.mp4',  help="Enter the output file directory")

    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()

    data = args.data
    annotations = args.annotations
    object_frame = args.object_frame
    if not exists(data) and exists(annotations):
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

    annotation, width, height, valid_frame = get_annotations(annotations,object_frame)
    center = (int((annotation['xmin']+annotation['xmax'])/2), int((annotation['ymin']+annotation['ymax'])/2))
    
    old_frame = cv2.imread( join(data,object_frame) )
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    lk_params = dict( winSize  = (25,25), maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    point_selected = True
    old_points = np.array([[center[0], center[1]]], dtype=np.float32)
    mask = np.zeros_like(old_frame)

    frm_no, cen_err = [], []
    all_frames = sorted(os.listdir(data))
    for i, frame in enumerate( all_frames[all_frames.index(object_frame):] ):
        #if i%2 == 0: continue
        frm_path = join(data,frame)
        print(frm_path)
        new_frame = cv2.imread(frm_path)
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY) 
        new_points, status, error = cv2.calcOpticalFlowPyrLK( old_gray, new_gray, old_points, None, **lk_params )
        old_gray = new_gray.copy()
        old_points = new_points
        a,b = old_points.ravel()
        c,d = new_points.ravel()
        mask = cv2.line(mask, (int(a),int(b)), (int(c),int(d)), (0,0,255), 4)
        cv2.circle(new_frame, (int(c),int(d)), 5, (0,0,255), 2)
        img = cv2.add(new_frame, mask)

        annotation, width, height, valid_frame = get_annotations(annotations,frame)
        center_error = None
        if valid_frame:
            center = (int((annotation['xmin']+annotation['xmax'])/2), int((annotation['ymin']+annotation['ymax'])/2))
            cv2.circle(img, center, 5, (0,255,0), 2)
            center_error = round( np.linalg.norm([(center[0]-int(c)), (center[1]-int(d))]), 3)

        cv2.putText(img, 'frm_no : '+str(i), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(img, 'center_err : '+str(center_error), (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        frm_no.append(i)
        cen_err.append(center_error)
        ax.plot(frm_no,cen_err,'g')
        frms = []
        for _ in range(i):frms.append(35)
        ax.plot(np.arange(i),frms, 'b')
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        image_from_plot = cv2.resize(image_from_plot,(600,450))
        img = cv2.resize(img,(800,450))
        concat_img = cv2.hconcat([img, image_from_plot])

        if args.save : output.write(concat_img) 
        cv2.imshow('video',concat_img)
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

    if args.save : output.release()
    cv2.destroyAllWindows()

