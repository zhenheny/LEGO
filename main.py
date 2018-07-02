from __future__ import division
import os
import sys
import time
sys.path.append("../eval")
import numpy as np
import scipy.misc
import tensorflow as tf
import argparse
from SfMLearner import SfMLearner
from evaluate_kitti import *
from evaluate_normal import *
from utils import *


def test_image(filename):

    mode = 'depth'
    img_height=256
    img_width=832
    ckpt_file = 'models/model-145248'
    # ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/sfmlearner/chpts/model.latest'
    # ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/d2nn2d_1pt/depth2normal_test/model-62875'
    ckpt_file = '/home/zhenheng/Datasets_4T/tf_events/unsp_depth_normal/cs_4pt0.0_noflyout_dilated2_d2nnei3_n2dedgeremove_depthsmooth_noedge_wedgel2_alpha10_clip0_wt4_normal_smooth_wt0.05_edge_lossscalefactor_input417_l2_deconvk4_nnupsample_noscaling_wt0.2_expwt0.8_sfmpy0723_depth1normal_eval_cont/model-100002'
    # img_path = "/home/zhenheng/Datasets_ssd/Datasets/kitti/training/image_2/"
    img_path = "/home/zhenheng/Documents/"
    # I = scipy.misc.imread('misc/sample.png')
    I = scipy.misc.imread(img_path+filename)
    I = scipy.misc.imresize(I, (img_height, img_width))

    sfm = SfMLearner()
    with tf.variable_scope("training"):
        sfm.setup_inference(img_height, img_width, mode=mode)

    saver = tf.train.Saver([var for var in tf.trainable_variables()])
    intrinsic = [[img_width, img_height, 0.5*img_width, 0.5*img_height]]
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4) 
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver.restore(sess, ckpt_file)
        pred = sfm.inference(I[None,:,:,:], intrinsic, sess, mode=mode)
    # np.save("./visualization.npy",normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.figure()
    # plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.savefig("./visualization.png", bbox_inches="tight")
    # plt.figure(figsize=(15,15))
    # plt.subplot(1,2,1); plt.imshow(I)
    # plt.subplot(1,2,2); plt.imshow(normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # plt.savefig("./visualization.pdf", bbox_inches="tight")
    scipy.misc.imsave("./visualize_depth.png", normalize_depth_for_display(pred['depth'][0,:,:,0]))
    # np.save("./1.npy", pred['disp'])

def test_filelist(filelist, split, eval_bool, ckpt_file):
    intrinsic_matrixes = []
    if split == "kitti":
        intrinsic_matrixes = pickle.load(open("/home/zhenheng/datasets/kitti/intrinsic_matrixes.pkl", "rb"))
    
    # save_path = "/home/zhenheng/Works/SfMLearner/eval/kitti/"
    if split in ['kitti', 'eigen']:
        root_img_path = "/home/zhenheng/datasets/kitti/"
        normal_gt_path = "/home/zhenheng/works/unsp_depth_normal/depth2normal/eval/kitti/gt_nyu_fill_depth2nornmal_tf/"
        normal_gt_path = "/home/zhenheng/works/unsp_depth_normal/depth2normal/eval/kitti/gt_nyu_fill_depth2nornmal_tf_mask/"
        normal_gt_path = "/home/zhenheng/datasets/kitti/"+split+"_normal_gt_monofill_mask/"
        test_fn = root_img_path+"test_files_"+split+".txt"
    elif split == "cs":
        root_img_path = "/home/zhenheng/datasets/cityscapes/"
        test_fn = root_img_path+"test_files_"+split+".txt"
        normal_gt_path = ""
    elif split == "make3d":
        root_img_path = "/home/zhenheng/datasets/make3d/"
        test_fn = root_img_path+"test_files_"+split+".txt"
        normal_gt_path = ""

    mode = 'depth'
    img_height=256
    img_width=832
    
    # ckpt_file = '/home/zhenheng/Datasets_4T/unsp_depth_normal/d2nn2d_1pt_new/model-40249'
    # ckpt_file = 'models/model-145248'
    sfm = SfMLearner()
    with tf.variable_scope("training"):
    	sfm.setup_inference(img_height, img_width, mode=mode)

    saver = tf.train.Saver([var for var in tf.trainable_variables()]) 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # with tf.name_scope("training"):
        saver.restore(sess, ckpt_file)
        pred_depths_test, pred_normals_test, pred_depths2_test = [], [], []
        for i, file in enumerate(filelist):
            resize_ratio = img_height / 375.0
            if intrinsic_matrixes != []:
                intrinsic = np.expand_dims(np.array(intrinsic_matrixes[file.split("/")[-1].split("_")[0]])[[0,4,2,5]] * resize_ratio, axis=0)
            else:
                if split == "kitti":
                    intrinsic = [[img_width, img_height, 0.5*img_width, 0.5*img_height]]
                elif split == "cs":
                    intrinsic = [[900.0, 756.0, 445.0, 172.0]]
                else:
                    intrinsic = [[img_width, img_height, 0.5*img_width, 0.5*img_height]]
            I = scipy.misc.imread(file)
            I = scipy.misc.imresize(I, (img_height, img_width))

            pred = sfm.inference(I[None,:,:,:], intrinsic, sess, mode=mode)
            # pred = sfm.inference(I[None,:,:,:], sess, mode=mode)
            pred_normal_np = np.squeeze(pred['normals'])
            # pred_normal_np[:,:,0], pred_normal_np[:,:,2] = pred_normal_np[:,:,2], pred_normal_np[:,:,0] 
            # pred_normal_np[:,:,0] *= -1
            # pred_normal_np[:,:,1] *= -1
            # pred_normal_np[:,:,2] *= -1
            # # pred_normal_np[:,:,0] -= 2
            # # pred_normal_np[:,:,2] -= 2
            # pred_normal_np = (pred_normal_np + 1.0) / 2.0

            # pred = sfm.inference(I[None,:,:,:], [], sess, mode=mode)
            pred_depths_test.append(pred['depth'][0,0:,0:,0])
            pred_depths2_test.append(pred['depth2'][0,5:-5,5:-5,0])
            # for s in range(4):
            #     print (pred['edges'][s].shape)
            #     edge_image = scipy.misc.imresize(np.squeeze(pred['edges'][s]), [img_height, img_width], interp="nearest")
            #     scipy.misc.imsave("../edge_vis/%03d_%01d.png" % (i, s), edge_image)
            # scipy.misc.imsave("./test_eval/%06d_10.png" % i, normalize_depth_for_display(pred['depth'][0,:,:,0]))
            pred_normals_test.append(pred_normal_np)
            # print(pred['edges'][0].shape)
            # scipy.misc.imsave("../eval/edge_asap_cs/%03d.jpg" % i, np.squeeze(pred['edges'][0]))
            scipy.misc.imsave("/home/zhenheng/datasets/cityscapes/sequences_vis/sequence10/edge/%03d.jpg" % i, np.squeeze(pred['edges'][0])[:-6,:,])

    if eval_bool:
        gt_depths, pred_depths, gt_disparities = load_depths(pred_depths_test, split, root_img_path, test_fn)
        eval_depth(gt_depths, pred_depths, gt_disparities, split, vis=True)
        # gt_depths, pred_depths2, gt_disparities = load_depths(pred_depths2_test, split, root_img_path, test_fn)
        # eval_depth(gt_depths, pred_depths2, gt_disparities, split, vis=False)
        pred_normals, gt_normals = load_normals(pred_normals_test, split, normal_gt_path,test_fn) 
        eval_normal(pred_normals, gt_normals, split, vis=True)
        # scipy.misc.imsave(save_path+"visualization/"+file.split("/")[-1], normalize_depth_for_display(pred['depth'][0,:,:,0]))
        # np.save(save_path+"npy_files/sfmlearner_depth.npy",npyfile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
    parser.add_argument('--split', type=str, help='data split, kitti or eigen', required=True)
    parser.add_argument('--gpu_id', type=str, help='gpu id for evaluation', default="0")
    parser.add_argument('--ckpt_file', type=str, help='model checkpoint', required=True, default='models/model-145248')
    parser.add_argument('--type',  type=str, help='test type, img or filelist', default='filelist')
    parser.add_argument('--eval_depth_bool', type=bool, help="evaluate the depth estimation based on standard metrics", default=False)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    if args.type == "filelist":
        filelist = []
        if args.split in ["kitti", "eigen"]:
        ## kitti
            test_file_list = "/home/zhenheng/datasets/kitti/test_files_"+args.split+".txt"
            with open(test_file_list) as f:
                for line in f:
                    filelist.append("/home/zhenheng/datasets/kitti/"+line.rstrip())
        elif args.split == "cs":
            # test_file_list = "/home/zhenheng/datasets/cityscapes/test_files_"+args.split+".txt"
            test_file_list = "/home/zhenheng/datasets/cityscapes/sequences_vis/sequence10/test_files_sequence10.txt"
            with open(test_file_list) as f:
                for line in f:
                    filelist.append("/home/zhenheng/datasets/cityscapes/"+line.rstrip())
        elif args.split == "make3d":
            test_file_list = "/home/zhenheng/datasets/make3d/test_files_"+args.split+".txt"
            with open(test_file_list) as f:
                for line in f:
                    filelist.append("/home/zhenheng/datasets/make3d/test_imgs/"+line.rstrip())
        test_filelist(filelist, args.split, args.eval_depth_bool, args.ckpt_file)
    else:
        filename = "example_detect.png"
        filename = '2.png'
        test_image(filename)
