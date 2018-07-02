from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob

class nyuv2_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split='train',
                 crop_bottom=False, # Get rid of the car logo
                 scene_num=4,
                 img_height=171, 
                 img_width=416,
                 seq_length=5,
                 sample_gap=5,
                 subset=""):  # Sample every two frames to match KITTI frame rate
        self.dataset_dir = dataset_dir
        self.split = split
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.sample_gap = sample_gap
        self.subset = subset
        self.scene_num = scene_num
        assert seq_length % 2 != 0, 'seq_length must be odd!'
        self.frames = self.collect_frames(split)
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)
        
    def collect_frames(self, split):
        counter = {}
        img_dir = self.dataset_dir + '/data/'
        # scene_list = os.listdir(img_dir)
        # scene_list = np.loadtxt(self.dataset_dir+"test_scenes.txt", dtype="str")
        subset = self.subset
        scene_list = glob(img_dir + subset+"*")
        frames = []
        for scene in scene_list:
            # if scene.split("_")[0] not in counter:
            #     counter[scene.split("_")[0]] = 1
            # else: counter[scene.split("_")[0]] += 1
            # if counter[scene.split("_")[0]] > self.scene_num:
            #     continue
            img_files = glob(img_dir + scene.split("/")[-1] + '/*.ppm')
            img_files.sort()
            for f in img_files:
                frames.append(f)
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_example(tgt_idx):
            return False
        example = self.load_example(tgt_idx)
        return example

    def load_intrinsics(self, frame_id, split):
        fx = 5.1885790117450188e+02
        fy = 5.1946961112127485e+02
        cx = 3.2558244941119034e+02
        cy = 2.5373616633400465e+02
        intrinsics = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0,  0,  1]])
        return intrinsics

    def is_valid_example(self, tgt_idx):
        tgt_frame_fn = self.frames[tgt_idx]
        tgt_scene = tgt_frame_fn.split("/")[-2]
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            if tgt_idx + o >= len(self.frames):
                return False
            curr_frame_fn = self.frames[tgt_idx + o]
            curr_scene = curr_frame_fn.split("/")[-2]
            if curr_scene != tgt_scene:
                return False
            if not os.path.exists(curr_frame_fn):
                return False
        return True

    def load_image_sequence(self, tgt_idx, seq_length, crop_bottom):
        half_offset = int((seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_frame_fn = self.frames[tgt_idx + o]
            curr_img = scipy.misc.imread(curr_frame_fn)
            if curr_img.shape == ():
                return False, [], []
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height/raw_shape[0]
                zoom_x = self.img_width/raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y
    
    def load_example(self, tgt_frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length, self.crop_bottom)
        if image_seq == False:
            return False
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        frame_fn = self.frames[tgt_frame_id]
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = frame_fn.split('/')[-2]
        example['file_name'] = frame_fn.split("/")[-1].split(".ppm")[0]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out