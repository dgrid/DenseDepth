import numpy as np
from utils import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from keras.backend import print_tensor
from augment import BasicPolicy
from pathlib import Path
import csv
import os
import cv2
import sys

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    name = {name: input_zip.read(name) for name in input_zip.namelist()}
    return name

##########################
# own dataset
##########################

def own_resize(img, resolution_H=480, resolution_W=640, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution_H, resolution_W), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_own_data(batch_size, data_dir, shape_rgb_2d, shape_depth_2d, train_csv_path, test_csv_path, debug=False):
    data_dir = Path(data_dir)
    with open(os.path.join(data_dir, train_csv_path), 'r') as train_f:
        reader = csv.reader(train_f, delimiter=',')
        own2_train = [row for row in reader]
    with open(os.path.join(data_dir, test_csv_path), 'r') as test_f:
        reader = csv.reader(test_f, delimiter=',')
        own2_test = [row for row in reader]

    own2_train_list = []
    for item in own2_train:
        own2_train_list.extend(item)
    own2_test_list = []
    for item in own2_test:
        own2_test_list.extend(item)
    #data = list(map(str, data_dir.glob("**/*")))
    #data = {str(f): f.read_bytes() for f in data_dir.glob("**/*") if f.is_file()}
    data = {}
    for f in data_dir.glob("**/*"):
        if f.is_file() and (str(f) in own2_train_list or str(f) in own2_test_list):
            data[str(f)] = f.read_bytes()

    shape_rgb = (batch_size, ) + shape_rgb_2d + (3, )
    shape_depth_reduced = (batch_size, ) + tuple([int(s / 2) for s in shape_depth_2d]) + (1, )

    # Helpful for testing...
    if debug:
        own2_train = own2_train[:10]
        own2_test = own2_test[:10]

    return data, own2_train, own2_test, shape_rgb, shape_depth_reduced

def get_own_train_test_data(batch_size, data_dir, shape_rgb_2d, shape_depth_2d, train_csv_path, test_csv_path, debug=False):
    data, own2_train, own2_test, shape_rgb, shape_depth_reduced = get_own_data(batch_size, data_dir, shape_rgb_2d, shape_depth_2d, train_csv_path, test_csv_path, debug)

    train_generator = own_BasicAugmentRGBSequence(data, own2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth_reduced=shape_depth_reduced)
    test_generator = own_BasicRGBSequence(data, own2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth_reduced=shape_depth_reduced)

    return train_generator, test_generator

class own_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth_reduced, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth_reduced = shape_depth_reduced
        self.orig_shape_depth = (batch_size, ) + tuple([int(s * 2) for s in shape_depth_reduced[1:3]]) + (1, )
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth_reduced )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)
            
            sample = self.dataset[index]
            #x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(self.shape_rgb[1:])/255,0,1)
            x = np.clip(cv2.resize(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )), self.shape_rgb[1:3]).reshape(self.shape_rgb[1:])/255,0,1)
            #x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) ))/255,0,1)
            
            #y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(self.orig_shape_depth[1:])/255*self.maxDepth,0,self.maxDepth)
            #y = np.clip(np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), 1 )).reshape(self.orig_shape_depth[1:])/255*self.maxDepth,0,self.maxDepth)
            #y = np.clip(cv2.resize(np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), 1 )), self.orig_shape_depth[1:3]).reshape(self.orig_shape_depth[1:])/255*self.maxDepth,0,self.maxDepth)
            #y=np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), 1 ))
            y = cv2.resize(np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), -1 )), (self.orig_shape_depth[2], self.orig_shape_depth[1]))
            y = np.clip(y[:, :, np.newaxis], 0, self.maxDepth)
            #y = np.clip(np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), 1 ))/255*self.maxDepth,0,self.maxDepth)
            y[y==0] = self.maxDepth
            y = DepthNorm(y, maxDepth=self.maxDepth)
            batch_x[i] = own_resize(x, self.shape_rgb[1], self.shape_rgb[2])
            batch_y[i] = own_resize(y, self.shape_depth_reduced[1], self.shape_depth_reduced[2])

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class own_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth_reduced):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth_reduced = shape_depth_reduced
        self.orig_shape_depth = (batch_size, ) + tuple([int(s * 2) for s in shape_depth_reduced[1:3]]) + (1, )
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth_reduced )
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            #x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(self.shape_rgb[1:])/255,0,1)
            x = np.clip(cv2.resize(np.asarray(Image.open( BytesIO(self.data[sample[0]]))), self.shape_rgb[1:3]).reshape(self.shape_rgb[1:])/255,0,1)
            #y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(self.orig_shape_depth[1:]).copy().astype(float) / 10.0
            #y = np.asarray(cv2.imdecode( np.fromstring(BytesIO(self.data[sample[1]]).read(),np.uint8),1 ), dtype=np.float32).reshape(self.orig_shape_depth[1:]).copy().astype(float) / 10.0
            #y = cv2.resize(np.asarray(cv2.imdecode( np.fromstring(BytesIO(self.data[sample[1]]).read(),np.uint8),1 ), dtype=np.float32), self.orig_shape_depth[1:3]).reshape(self.orig_shape_depth[1:]).copy().astype(float) / 10.0
            y = cv2.resize(np.asarray(cv2.imdecode( np.fromstring( BytesIO(self.data[sample[1]]).read() , np.uint8), -1 )), (self.orig_shape_depth[2], self.orig_shape_depth[1]))
            y = np.clip(y[:, :, np.newaxis], 0, self.maxDepth)
            y[y==0] = self.maxDepth
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = own_resize(x, self.shape_rgb[1], self.shape_rgb[2])
            batch_y[i] = own_resize(y, self.shape_depth_reduced[1], self.shape_depth_reduced[2])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

##########################
# NYU dataset
##########################
def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip', debug=False):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if debug:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size, debug):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size, debug=debug)

    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(480,640,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(480,640,3)/255,0,1)
            y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

#================
# Unreal dataset
#================

import cv2
from skimage.transform import resize

def get_unreal_data(batch_size, unreal_data_file='unreal_data.h5'):
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Open data file
    import h5py
    data = h5py.File(unreal_data_file, 'r')

    # Shuffle
    from sklearn.utils import shuffle
    keys = shuffle(list(data['x'].keys()), random_state=0)

    # Split some validation
    unreal_train = keys[:len(keys)-100]
    unreal_test = keys[len(keys)-100:]

    # Helpful for testing...
    if False:
        unreal_train = unreal_train[:10]
        unreal_test = unreal_test[:10]

    return data, unreal_train, unreal_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)

    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2,
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            depth_sample = self.data['y/{}'.format(sample)]
            depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )

            x = np.clip(rgb_sample/255, 0, 1)
            y = np.clip(depth_sample, 10, self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y
