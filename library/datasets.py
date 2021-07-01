import os
from os.path import join, isfile
from math import sqrt
import random
import json
import numpy as np
from imageio import imread
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from library.exceptions import IncompatibleFolderStructure
from library.train_utils import get_annotations, check_folder_tree
from library.labels import create_BCELogit_loss_label
from library.crops_train import crop_img, resize_and_pad

class MyDataset(Dataset):
    
    def __init__(self, dataset_dir, dataset_type=None, reference_size=127, search_size=255, final_size=33,
                 upscale_factor=4, max_frame_sep=50, pos_thr=25, neg_thr=50,
                 cxt_margin=0.5, metadata_file=None, save_metadata=None, transforms=ToTensor()):
        
        if not check_folder_tree(dataset_dir,dataset_type):
            raise IncompatibleFolderStructure
        
        self.dataset = join(dataset_dir,dataset_type)
        self.reference_size = reference_size
        self.search_size = search_size
        self.final_size = final_size

        self.upscale_factor = upscale_factor
        self.max_frame_sep = max_frame_sep
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        
        self.cxt_margin = cxt_margin       
        self.transforms = transforms
                       
        self.get_metadata(metadata_file, save_metadata)

    def get_metadata(self, metadata_file, save_metadata):
        if metadata_file and isfile(metadata_file):
            with open(metadata_file) as json_file:
                mdata = json.load(json_file)
            if self.check_metadata(mdata):
                print("Metadata file found. Loading its content.")
                for key, value in mdata.items():
                    setattr(self, key, value)
                return
        mdata = self.build_metadata()
        if save_metadata is not None:
            with open(save_metadata, 'w') as outfile:
                json.dump(mdata, outfile)
    def check_metadata(self, metadata):
        if not all(key in metadata for key in ('frames', 'annotations', 'list_idx')) : return False
        if not (isfile(metadata['frames'][0][0]) and isfile(metadata['frames'][-1][-1])) : return False
        return True
    def build_metadata(self):
        frames = []
        annotations = []
        list_idx = []

        for i,scene in enumerate(sorted(os.listdir(self.dataset))):
            datadir = join(self.dataset,scene,'data')
            annodir = join(self.dataset,scene,'annotations')

            seq_frames = []
            seq_annots = []

            for frame in sorted(os.listdir(datadir)):
                annot, h, w, valid = get_annotations(annodir, frame)
                if valid:
                    seq_frames.append(join(datadir, frame))
                    seq_annots.append(annot)
                    list_idx.append(i)

            frames.append(seq_frames)
            annotations.append(seq_annots)

        metadata = {'frames': frames, 'annotations': annotations, 'list_idx': list_idx}
        for key, value in metadata.items():
            setattr(self, key, value)
        return metadata

    def __len__(self):
        return len(self.list_idx)

    def __getitem__(self, idx):
        seq_idx = self.list_idx[idx]
        first_idx, second_idx = self.get_pair(seq_idx)
        return self.preprocess_sample(seq_idx, first_idx, second_idx)
        
    def get_pair(self, seq_idx, frame_idx=None):
        size = len(self.frames[seq_idx])
        if frame_idx is None:
            first_frame_idx = random.randint(0, size-1)
        else:
            first_frame_idx = frame_idx
        min_frame_idx = max(0, (first_frame_idx - self.max_frame_sep))
        max_frame_idx = min(size - 1, (first_frame_idx + self.max_frame_sep))
        second_frame_idx = random.randint(min_frame_idx, max_frame_idx)
        
        return first_frame_idx, second_frame_idx
    def preprocess_sample(self, seq_idx, first_idx, second_idx):
        reference_frame_path = self.frames[seq_idx][first_idx]
        search_frame_path = self.frames[seq_idx][second_idx]
        #print(reference_frame_path,search_frame_path)
        ref_annot = self.annotations[seq_idx][first_idx]
        srch_annot = self.annotations[seq_idx][second_idx]

        ref_w = (ref_annot['xmax'] - ref_annot['xmin'])/2
        ref_h = (ref_annot['ymax'] - ref_annot['ymin'])/2

        ref_ctx_size = self.ref_context_size(ref_h, ref_w)
        ref_cx = (ref_annot['xmax'] + ref_annot['xmin'])/2
        ref_cy = (ref_annot['ymax'] + ref_annot['ymin'])/2
        
        ref_frame = imread(reference_frame_path)
        ref_frame = np.float32(ref_frame)

        ref_frame, pad_amounts_ref = crop_img(ref_frame, ref_cy, ref_cx, ref_ctx_size)
        try:
            ref_frame = resize_and_pad(ref_frame, self.reference_size, pad_amounts_ref, reg_s=ref_ctx_size)
        except AssertionError:
            print('Fail Ref: ', reference_frame_path)
            raise
        
        srch_ctx_size = ref_ctx_size * self.search_size / self.reference_size
        srch_ctx_size = (srch_ctx_size//2)*2 + 1

        srch_cx = (srch_annot['xmax'] + srch_annot['xmin'])/2
        srch_cy = (srch_annot['ymax'] + srch_annot['ymin'])/2
        
        srch_frame = imread(search_frame_path)
        srch_frame = np.float32(srch_frame)
        srch_frame, pad_amounts_srch = crop_img(srch_frame, srch_cy, srch_cx, srch_ctx_size)
        try:
            srch_frame = resize_and_pad(srch_frame, self.search_size, pad_amounts_srch, reg_s=srch_ctx_size)
        except AssertionError:
            print('Fail Search: ', search_frame_path)
            raise
        
        ref_frame = self.transforms(ref_frame)
        srch_frame = self.transforms(srch_frame)
        label = create_BCELogit_loss_label(self.final_size, self.pos_thr, self.neg_thr, upscale_factor=self.upscale_factor)
        out_dict = {'ref_frame': ref_frame, 'srch_frame': srch_frame, 'label': label,
                    'seq_idx': seq_idx, 'ref_idx': first_idx, 'srch_idx': second_idx }
        return out_dict
    def ref_context_size(self, h, w):
        margin_size = self.cxt_margin*(w + h)
        ref_size = sqrt((w + margin_size) * (h + margin_size))
        # make sure ref_size is an odd number
        ref_size = (ref_size//2)*2 + 1
        return int(ref_size)


    

    
