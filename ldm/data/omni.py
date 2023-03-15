import pdb
import os
import os.path as osp
from collections import defaultdict
import io
import tarfile
import tarfile
import sys
import os
import time
import codecs

import json
import pickle

import random
import albumentations
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
starttime = time.time()

class OmniBase(Dataset):
    def __init__(self,
                 txt_file,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.mesh_root = '/nfs/turbo/fouheyTemp/jinlinyi/datasets/gibson'
        self.tar_root = '/nfs/turbo/fouheyTemp/jinlinyi/datasets/omnidata/compressed'
        self.index_root = '/nfs/turbo/fouheyTemp/relh/latent-diffusion/data/omni/omniindex/'
        self.extracted_root = '/nfs/turbo/fouheyTemp/jinlinyi/datasets/omnidata/omnidata_taskonomy'
        self.pose_cached_file = '/nfs/turbo/fouheyTemp/nileshk/mv_drdf_cachedir/taskonomy_gathered_point_info'

        self.data_paths = txt_file
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self.image_paths = [x for x in self.image_paths if 'cottonport/point_2772_view_0_domain' not in x]
        self.shuffle()

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS, }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)

        self.house_ids = list(set([x.split('/')[0] for x in self.image_paths]))
        self.all_tars = defaultdict(lambda: defaultdict(object))
        self.all_indices = defaultdict(lambda: defaultdict(object))
        for house_id in self.house_ids:
            for data_type in ['rgb', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'mask_valid']:
                tar_name = (
                    osp.join(self.tar_root, f"{data_type}__taskonomy__{house_id}.tar")
                )
                index_name = (
                    osp.join(self.index_root, f"{data_type}__taskonomy__{house_id}.tar")
                )

                self.all_tars[house_id][data_type] = open(tar_name, 'rb')
                self.all_indices[house_id][data_type] = open(index_name, 'r')#.readlines()
        # splits = ['gibson_full', 'gibson_fullplus', 'gibson_medium', 'gibson_tiny', 'gibson_v2']

    def get_house_ids(self):
        return self.house_ids

    def get_house_mesh_by_house_id(self, house_id):
        mesh_path = osp.join(self.mesh_root, self.id2split[house_id], house_id.capitalize(), 'mesh_z_up.obj')
        with open(mesh_path, "rb") as f:
            mesh = trimesh.exchange.obj.load_obj(f, include_color=True)
        return mesh

    def get_all_dp_ids_by_house_id(self, house_id):
        if house_id not in self.house_id2dp_id.keys():
            suffix = "_rgb.png"
            dp_ids = [
                f.split(suffix)[0]
                for f in os.listdir(
                    osp.join(self.extracted_root, f"rgb/taskonomy/{house_id}")
                )
            ]
            self.house_id2dp_id[house_id] = dp_ids
        return self.house_id2dp_id[house_id]

    def get_all_cameras_by_house_id(self, house_id):
        with open(osp.join(self.pose_cached_file, f'{house_id}.json')) as f:
            data = json.load(f)
        return data

    def lookup(self, path, house_id, data_type):
        if hasattr(self.all_indices[house_id][data_type], 'readlines'):
            self.all_indices[house_id][data_type] = self.all_indices[house_id][data_type].readlines()

        for line in self.all_indices[house_id][data_type]:
            m = line[:-1].rsplit(" ", 2)
            if path == m[0]:
                self.all_tars[house_id][data_type].seek(int(m[1]))
                buffer = self.all_tars[house_id][data_type].read(int(m[2]))
                return buffer
                #os.write (sys.stdout.fileno (), buffer)

    def get_from_tar(self, house_id, dp_id, data_type):
        if data_type == 'mask_valid':
            data_name = f"./mask_valid/{house_id}/{dp_id}_depth_zbuffer.png"
        elif data_type == 'point_info':
            data_name = f"point_info/{dp_id}_point_info.json"
        else:
            data_name = f"{data_type}/{dp_id}_{data_type}.png"

        #try:
        if True:
            #print(data_type)
            #print(house_id)
            #print(dp_id)
            #data = self.all_tars[house_id][data_type].extractfile(data_name)
            data = self.lookup(data_name, house_id, data_type)
            if data_type != "point_info":
                #data = data.read()
                data = Image.open(io.BytesIO(data))
                #data = get_transform(data_type)(data)
            else:
                data = json.loads(data)#.read())
        #except:
        #    breakpoint()
        return data

    def __len__(self):
        return self._length

    def crop(self, x, size=640):
        x = x[(x.shape[0] - size) // 2:-(x.shape[0] - size) // 2,\
              (x.shape[1] - size) // 2:-(x.shape[1] - size) // 2]
        return x

    def shuffle(self):
        random.shuffle(self.image_paths)
        self.this_image_paths = self.image_paths[:1000]
        self._length = len(self.this_image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.this_image_paths],
            "file_path_": [os.path.join(self.tar_root, l)
                           for l in self.this_image_paths],
        }

    def __getitem__(self, i):
        if i == 0:
            self.shuffle()

        example = dict((k, self.labels[k][i]) for k in self.labels)
        #print(self.image_paths[i])
        house_id, dp_id = self.this_image_paths[i].split('/')

        try:
            rgb = self.get_from_tar(house_id, dp_id, 'rgb') # 3
            depth = self.get_from_tar(house_id, dp_id, 'depth_zbuffer') # 1
            normal = self.get_from_tar(house_id, dp_id, 'normal') # 3
            curvature = self.get_from_tar(house_id, dp_id, 'principal_curvature') # 3
            reshading = self.get_from_tar(house_id, dp_id, 'reshading') # 3
            mask_valid = self.get_from_tar(house_id, dp_id, 'mask_valid') # 1
            #point_info = self.get_from_tar(house_id, dp_id, 'point_info')
        except:
            return self[random.randint(0, len(self))]

        to_np = lambda x : (np.array(x).astype(np.uint8) / 127.5 - 1.0).astype(np.float32)
        np_rgb = to_np(rgb)
        np_depth = to_np(depth)[:, :, None]
        np_normal = to_np(normal)
        np_curvature = to_np(curvature)
        np_reshading = to_np(reshading)
        np_mask_valid = to_np(mask_valid)[:, :, None]

        image = np.concatenate((
            np_rgb,
            np_depth,
            np_normal,
            np_curvature,
            np_reshading,
            np_mask_valid
        ), axis=2)

        example["class"] = np_rgb #self.crop(np_rgb, size=512)
        example["image"] = image #self.crop(image, size=512)
        return example

class OmniTrain(OmniBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/omni/omni_train_exists.txt", **kwargs)

class OmniValidation(OmniBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/omni/omni_valid_exists.txt",
                         flip_p=flip_p, **kwargs)

def make_valid(split='train'):
    tar_root = 'data/omni/omniindex/'
    all_files = open(f'data/omni/omni_{split}.txt').readlines()
    house_ids = list(set([x.split('/')[0] for x in all_files]))

    existing_samples = []
    #for f in all_files:
    for house_id in house_ids:
        tar_f = {}
        for data_type in ['rgb', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'mask_valid']:
            tar_name = (
                osp.join(tar_root, f"{data_type}__taskonomy__{house_id}.tar")
            )

            try:
                tar_f[data_type] = tarfile.open(tar_name, 'r|')
            except: continue 

        # make RGB the primary key
        these_houses = defaultdict(int) 
        for data_type in ['rgb', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'mask_valid']:
            if data_type not in tar_f: continue

            #data = tar_f[data_type].next()
            #if '.png' not in data.name:
            #    data = tar_f[data_type].next()
            for data in tar_f[data_type]:
                if '.png' not in data.name:
                    continue

                if data_type == 'depth_zbuffer' or data_type == 'principal_curvature' or data_type == 'mask_valid':
                    dp_id = '_'.join(data.name.split('/')[-1].split('_')[:-2])
                else:
                    dp_id = '_'.join(data.name.split('/')[-1].split('_')[:-1])

                these_houses[dp_id] += 1

        for dp_id, exists in these_houses.items():
            if exists == 6:
                existing_samples.append(house_id + '/' + dp_id)
        print(len(existing_samples))

    with open(f'data/omni/omni_{split}_exists.txt', 'a') as f:
        for eh in existing_samples:
            f.write(eh + '\n')


def make_all(split='train'):
    house_ids = list(dataloader.get_house_ids())
    house_id = 'brevort'
    #mesh = dataloader.get_house_mesh_by_house_id(house_id)
    #cameras = dataloader.get_all_cameras_by_house_id(house_id)
    #dp_ids = dataloader.get_all_dp_ids_by_house_id(house_id)
    print(len(house_ids))

    train_house_ids = house_ids[:500]
    val_house_ids = house_ids[500:]

    num_images = 0 
    for zzz, house_id in enumerate(train_house_ids):
        try: 
            print(house_id)
            dp_ids = dataloader.get_all_dp_ids_by_house_id(house_id)
            print(len(dp_ids))
            num_images += len(dp_ids)

            with open('data/omni/omni_train.txt', 'a') as f:
                for dp_id in dp_ids:
                    f.write(str(house_id) + '/' + str(dp_id) + '\n')
        except:
            pass

    for zzz, house_id in enumerate(val_house_ids):
        try: 
            print(house_id)
            dp_ids = dataloader.get_all_dp_ids_by_house_id(house_id)
            print(len(dp_ids))
            num_images += len(dp_ids)

            with open('data/omni/omni_valid.txt', 'a') as f:
                for dp_id in dp_ids:
                    f.write(str(house_id) + '/' + str(dp_id) + '\n')
        except:
            pass

    print(num_images)


def human(size):
    a = 'B','kB','mB','gB','tB','pB'
    curr = 'B'
    while( size > 1024 ):
        size/=1024
        curr = a[a.index(curr)+1]
    return str(int(size*10)/10)+curr
    
def indextar(dbtarfile,indexfile):
    filesize = os.path.getsize(dbtarfile)
    lastpercent = 0
    

    with tarfile.open(dbtarfile, 'r|') as db:
        if os.path.isfile(indexfile):
            print('file exists. exiting')

        with open(indexfile, 'w') as outfile:
            counter = 0
            print('One dot stands for 1000 indexed files.')
            #tarinfo = db.next()
            for tarinfo in db:
                currentseek = tarinfo.offset_data
                rec = "%s %d %d\n" % (tarinfo.name, tarinfo.offset_data, tarinfo.size)
                outfile.write(rec)
                counter += 1
                if counter % 1000 == 0:
                    # free ram...
                    db.members = []
                if(currentseek/filesize>lastpercent):
                    print('')
                    percent = int(currentseek/filesize*1000.0)/10
                    print(str(percent)+'%')
                    lastpercent+=0.01
                    print(human(currentseek)+'/'+human(filesize))
                    if(percent!=0):
                        estimate = ((time.time()-starttime)/percent)*100
                        eta = (starttime+estimate)-time.time()
                        print('ETA: '+str(int(eta))+'s (estimate '+str(int(estimate))+'s)')
    print('done.')


if __name__ == "__main__": 
    #make_valid(split='valid')
    #make_valid(split='train')

    #one_dataloader = OmniTrain()
    #one_dataloader[0]

    dataloader = OmniValidation()

    for i in range(100):
        dataloader.shuffle()
        for x in dataloader:
            print(i)
    pdb.set_trace()

    '''

    dataloader[0]
    split = 'valid'
    all_files = open(f'data/omni/omni_{split}_exists.txt').readlines()
    house_ids = list(set([x.split('/')[0] for x in all_files]))
    tar_root = '/nfs/turbo/fouheyTemp/jinlinyi/datasets/omnidata/compressed'

    these_houses = defaultdict(lambda: defaultdict(object))
    for house_id in house_ids:
        print(house_id)
        for data_type in ['rgb', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'mask_valid']:
            tar_name = (
                osp.join(tar_root, f"{data_type}__taskonomy__{house_id}.tar")
            )
            these_houses[house_id][data_type] = tarfile.open(tar_name, 'r')
        break
    '''

    '''
    root = '/nfs/turbo/fouheyTemp/relh/latent-diffusion/data/omni/omnidata/'
    for path in os.listdir(root):
        print(path)
        if not os.path.exists(root.replace('omnidata', 'omniindex') + path):
            MODE = 'index'
            dbtarfile = root + path
            indexfile = root.replace('omnidata', 'omniindex') + path
            try:
                indextar(dbtarfile,indexfile)
            except:
                print('broken!')
                continue
    '''

    #main()    

