import xml.etree.ElementTree as ET
import os
from os.path import join, isdir, isfile, splitext
import json
import shutil
import torch

class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    def update_with_dict(self, dictio):
        """ Updates the parameters with the keys and values of a dictionary."""
        self.__dict__.update(dictio)
    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total/float(self.steps)

def check_folder_tree(root_dir,dataset_type):
    datasetpath = join(root_dir,dataset_type)
    necessary_folders = []
    for scene in sorted(os.listdir(datasetpath)):
        scenedatapath = join(datasetpath,scene,'data')
        sceneannopath = join(datasetpath,scene,'annotations')
        necessary_folders.extend((sceneannopath,scenedatapath))

    return all(isdir(path) for path in necessary_folders)

def get_annotations(annot_dir, frame_file):
    frame_no = splitext(frame_file)[0]
    annot_path = join(annot_dir, frame_no + '.xml')
    
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

def save_checkpoint(state, is_best, checkpoint):
    lastpath = join(checkpoint, 'last.pth.tar')
    torch.save(state, lastpath)
    if is_best:
        bestpath = join(checkpoint, 'best.pth.tar')
        shutil.copyfile(lastpath, bestpath)

def save_dict_to_json(d, is_best, bestepoch, filepath):
    d['epoch'] = bestepoch
    lastpath = join(filepath, "last_metrics_values.json")
    with open(lastpath, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
    if is_best:
        best_path = join(filepath, "best_metrics_values.json")
        shutil.copyfile(lastpath,best_path)

def load_checkpoint(checkpoint, model, optimizer=None):
    checkpoint = torch.load(checkpoint) if torch.cuda.is_available() \
        else torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint

def load_json_to_dict(json_path):
    with open(json_path) as json_file:
        data = json.load(json_file)
    return data

