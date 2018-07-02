## functions for evaluating normals
## Zhenheng Yang 06/27/2017

import numpy as np
import scipy.misc as sm
import argparse
from evaluation_utils import *
import os
import math
import time
import numpy.linalg as nla
import skimage.transform as st

def load_normals(pred_normals, split, gt_path, test_fn):
	## pred normals can be a npy file containing all predicted normals
	## or it can be a path containing normal images
	## load gt_normals
	if split in ["kitti", "eigen"]:
		gt_normals = []
		gt_files = os.listdir(gt_path)
		gt_files.sort()
		for file in gt_files:
			if file[0] == ".": continue 
			gt_normals.append(transform(sm.imread(gt_path + file)))

	if split == "cs":
		num_samples = 500
		gt_normals = [[] for i in range(num_samples)]
	
	if split == "nyuv2":
		gt_normals = []
		with open(test_fn) as f:
			for line in f:
				gt_normal = sm.imread(gt_path + "normals/" + line.rstrip().split("/")[-1])
				gt_mask = np.tile(np.expand_dims(sm.imread(gt_path + "masks/" + line.rstrip().split("/")[-1]), -1), (1,1,3))
				gt_normals.append(transform(gt_normal * gt_mask))
		pred_normals, gt_normals = np.array(pred_normals, dtype=np.float32), np.array(gt_normals, dtype=np.float32)

	return pred_normals, gt_normals


def transform(normal_img):
	return (normal_img / 255.0 * 2.0 -1.0)

def normalize_norm(normal_maps, axis):
	## l2 normalize normal vector

	import numpy.linalg as nla
	norms = nla.norm(normal_maps, axis=axis)
	for i in range(normal_maps.shape[axis]):
		normal_maps[:,:,:,i] /= norms

	return normal_maps

def eval_normal(pred_normals, gt_normals, split):
	## normal evaluation core function
	## input are two numpy arrays with same shape [batch, width, height, 3]
	epsilon = 1e-5
	degrees = []

	for i in range(len(pred_normals)):
		pred_normal = pred_normals[i]
		if split != "cs":
			gt_normal = gt_normals[i]
			gt_normal_normed = gt_normal
			pred_normal = np.clip(pred_normal, -1.0, 1.0)
			pred_normal = st.resize(pred_normal, gt_normal.shape)
			pred_normal_normed = pred_normal
			
			masks = (gt_normal > -0.95)
			mask = masks[:,:,0] | masks[:,:,1] | masks[:,:,2]
			pred_normal_norm, gt_normal_norm = nla.norm(pred_normal, axis=2), nla.norm(gt_normal, axis=2)

			for j in range(pred_normal.shape[-1]):
				pred_normal_normed[:,:,j] /= (pred_normal_norm + 1e-6)
				gt_normal_normed[:,:,j] /= (gt_normal_norm + 1e-6)
			
			in_prod = np.sum(pred_normal_normed * gt_normal_normed, axis=-1)
			degree_map = np.arccos(in_prod) / math.pi
		else:
			continue

		degree = np.arccos(in_prod[mask])*180.0/math.pi
		degrees += list(degree)
	if split == "cs": return
	degrees = np.array(degrees)
	print ("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('mean', 'median', '11.25', '22.5', '30'))
	print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}".format(np.mean(degrees), np.median(degrees), \
			np.mean(degrees < 11.25), np.mean(degrees < 22.5), np.mean(degrees < 30)))
	return np.mean(degrees), np.median(degrees), np.mean(degrees < 11.25), np.mean(degrees < 22.5), np.mean(degrees < 30)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Normal Evaluation')
	parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',   default="kitti")
	parser.add_argument('--pred_normals_path', type=str,   help='path to estimated normals',      required=True)
	parser.add_argument('--gt_path',             type=str,   help='path to ground truth normals', required=True)
	
	args = parser.parse_args()

	split, pred_normals_path, gt_path = args.split, args.pred_normals_path, args.gt_path
	start_time = time.time()

	if pred_normals_path.split(".")[-1] == "npy":
		pred_normals = np.load(pred_normals_path)
	else:
		pred_normals = []
		pred_files = os.listdir(pred_normals_path)
		pred_files.sort()
		for file in pred_files:
			if file[0] == ".": continue
			pred_normals.append(transform(sm.imread(pred_normals_path + file)))

	pred_normals, gt_normals = load_normals(pred_normals, split, gt_path, test_fn = "")
	print (time.time() - start_time)
	eval_normal(pred_normals, gt_normals)