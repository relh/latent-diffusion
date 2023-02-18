import os
import pickle

import random
import albumentations
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MYSTBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        self.flow_root = data_root.replace('myst_images', 'myst_flow')
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self.image_paths = [x + '.png' for x in self.image_paths]

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS, }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        example_10 = dict((k, self.labels[k][i].replace('myst_images', 'myst_flow/10')) for k in self.labels)
        example_20 = dict((k, self.labels[k][i].replace('myst_images', 'myst_flow/20')) for k in self.labels)
        #example_30 = dict((k, self.labels[k][i].replace('myst_images', 'myst_flow/30')) for k in self.labels)

        try:
            flow_10 = np.load(example_10["file_path_"].replace('.png', '.npz'), allow_pickle=True, mmap_mode='r')['arr_0'] / 648.0 
            flow_20 = np.load(example_20["file_path_"].replace('.png', '.npz'), allow_pickle=True, mmap_mode='r')['arr_0'] / 648.0 
            #flow_30 = np.load(example_30["file_path_"].replace('.png', '.npz'), allow_pickle=True, mmap_mode='r')['arr_0'] / 648.0 
        except:
            print('missing!')
            return self[random.randint(0, len(self))]

        # todo get right flow along with images 
        image_0 = Image.open(example["file_path_"])
        frame_num = int(example["file_path_"].split('_')[-1].split('.')[0])
        image_10 = Image.open(example["file_path_"].replace(str(frame_num), str(frame_num+10)))
        image_20 = Image.open(example["file_path_"].replace(str(frame_num), str(frame_num+20)))
        #image_30 = Image.open(example["file_path_"].replace(str(frame_num), str(frame_num+30)))

        if not image_0.mode == "RGB":
            image_0 = image_0.convert("RGB")
            image_10 = image_10.convert("RGB")
            image_20 = image_20.convert("RGB")
            #image_30 = image_30.convert("RGB")

        image_0 = (np.array(image_0).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
        image_10 = (np.array(image_10).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
        image_20 = (np.array(image_20).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
        #image_30 = (np.array(image_30).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)

        image = np.concatenate((
            image_0,
            image_10,
            image_20,
            #image_30,
            flow_10,
            flow_20,
            #flow_30,
        ), axis=2)

        example["class"] = image_0
        example["image"] = image
        return example

class MYSTFakeTrain(MYSTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/myst/myst_fake_train.txt", data_root="data/myst/myst_images", **kwargs)

class MYSTTrain(MYSTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/myst/myst_train.txt", data_root="data/myst/myst_images", **kwargs)

class MYSTValidation(MYSTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/myst/myst_valid.txt", data_root="data/myst/myst_images",
                         flip_p=flip_p, **kwargs)

if __name__ == "__main__": 
    import pdb
    split = 'train'
    '''
    tf1 = set(pickle.load(open(f'/home/relh/inferring_actions/util/10_ego4d_{split}_frames.pkl', 'rb')))
    tf2 = set(pickle.load(open(f'/home/relh/inferring_actions/util/20_ego4d_{split}_frames.pkl', 'rb')))
    tf3 = set(pickle.load(open(f'/home/relh/inferring_actions/util/30_ego4d_{split}_frames.pkl', 'rb')))
    # most tf2 frames are in tf1
    int_1_2 = tf1.intersection(tf2)
    int_all = int_1_2.intersection(tf3)
    print(len(int_all))

    with open(f'data/myst/myst_{split}.txt', 'w') as f:
        for fn in int_all:
            f.write(fn.split('_')[0] + '/' + fn + '\n')
    '''
    myst_train = MYSTTrain()
    all_crops = []

    for i in range(100):
        data_zero = myst_train[i]
        all_crops.append(data_zero['image'].min())
        all_crops.append(data_zero['image'].max())
        #all_crops.append(data_zero["crop"])

    #print(min(all_crops))
    print(min(all_crops))
    print(max(all_crops))
    pdb.set_trace()
