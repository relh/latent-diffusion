import os
import math
import numpy as np
import albumentations
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AIAHMIBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        if txt_file is None:
            self.data_paths = os.listdir(data_root)
            self.image_paths = [x for x in self.data_paths if '.npz' in x]
        else:
            self.data_paths = txt_file
            with open(self.data_paths, "r") as f:
                self.image_paths = f.read().splitlines()

        self.data_root = data_root
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

    def crop(self, x):
        x = x[(x.shape[0] - 256) // 2:-(x.shape[0] - 256) // 2, (x.shape[1] - 256) // 2:-(x.shape[1] - 256) // 2]
        #print(x.shape)
        if x.shape[0] != 256:
            print(x.shape)
        if x.shape[1] != 256:
            print(x.shape)
        return x

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        npz = np.load(example['file_path_'], allow_pickle=True, mmap_mode='r')
        X = npz['X']
        Y = np.concatenate((npz['Y1'], npz['Y2']), axis=2)
        #AIA = npz['AIACalibration']

        #image = Image.open(example["file_path_"])
        #if not image.mode == "RGB":
        #    image = image.convert("RGB")

        # default to score-sde preprocessing
        #img = np.array(image).astype(np.uint8)
        #h, w, = img.shape[0], img.shape[1]
        #print(img.shape)
        #cls = img[:, :(w // 2)]
        #img = img[:, (w // 2):]
        X = self.crop(X)
        Y = self.crop(Y)

        #cropcls = self.cropper(image=cls)["image"]
        #image = self.cropper(image=img)["image"]

        #print(cls.shape)
        #print(img.shape)
        #crop = min(300, img.shape[0], img.shape[1])
        #h, w, = img.shape[0], img.shape[1]
        #cls = cls[(h - crop) // 2:(h + crop) // 2,
        #      (w - crop) // 2:(w + crop) // 2]
        #img = img[(h - crop) // 2:(h + crop) // 2,
        #      (w - crop) // 2:(w + crop) // 2]
        #print(cropcls.sum())
        #print(image.sum())

        #cropcls = Image.fromarray(cropcls)
        #image = Image.fromarray(image)
        #if self.size is not None:
        #    cropcls = cropcls.resize((self.size, self.size), resample=self.interpolation)
        #    image = image.resize((self.size, self.size), resample=self.interpolation)
        #image = self.flip(image)
        #print(cropcls)
        #print(image)

        #cropcls = np.array(cropcls).astype(np.uint8)
        #image = np.array(image).astype(np.uint8)
        #example["class"] = (cropcls / 127.5 - 1.0).astype(np.float32)
        #example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["class"] = (X).astype(np.float32)
        example["image"] = (Y).astype(np.float32)
        return example

class AIAHMITrain(AIAHMIBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file=None, data_root="data/aiahmi/aiahmi_images/train", **kwargs)

class AIAHMIValidation(AIAHMIBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file=None, data_root="data/aiahmi/aiahmi_images/val",
                         flip_p=flip_p, **kwargs)

if __name__ == "__main__": 
    import pdb
    aiahmi_train = AIAHMITrain()
    all_crops = []

    for i in range(10000):
        data_zero = aiahmi_train[i]
        #all_crops.append(data_zero["crop"])

    #print(min(all_crops))
    pdb.set_trace()
