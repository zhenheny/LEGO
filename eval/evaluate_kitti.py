import numpy as np
import scipy.misc as sm
import argparse
import matplotlib.pyplot as plt
from evaluation_utils import *


def load_depths(pred_depths_org, split, gt_path, test_fn):

    if split == 'kitti':
        num_samples = 200
        
        gt_disparities = load_gt_disp_kitti(gt_path)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_depths_org)

    if split == 'eigen':
        num_samples = 698
        test_files = read_text_lines(gt_path + 'test_files_eigen.txt')
        gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(test_files, gt_path)

        num_test = len(im_files)
        gt_depths = []
        pred_depths = []
        gt_disparities = [[] for i in range(num_samples)]
        for t_id in range(num_samples):
            camera_id = cams[t_id]  # 2 is left, 3 is right
            depth = generate_depth_map(gt_calib[t_id], gt_files[t_id], im_sizes[t_id], camera_id, False, True)
            gt_depths.append(depth.astype(np.float32))
            
            disp_pred = sm.imresize(pred_depths_org[t_id], (im_sizes[t_id][0], im_sizes[t_id][1]), interp="bilinear", mode='F')

            pred_depths.append(disp_pred)

    if split == "nyuv2":
        gt_depths = load_gt_disp_nyuv2(gt_path, test_fn)
        num_samples = len(gt_depths)
        pred_depths = []
        gt_disparities = [[] for i in range(num_samples)]
        for t_id in range(num_samples):
            pred_depth = sm.imresize(pred_depths_org[t_id], gt_depths[0].shape, mode="F")
            pred_depths.append(pred_depth)

    if split == "cs":
        num_samples = 500
        gt_depths = [[] for i in range(num_samples)]
        gt_disparities = [[] for i in range(num_samples)]
        pred_depths = pred_depths_org 

    if split == "make3d":

        gt_disparities = load_gt_disp_make3d(gt_path, test_fn)
        gt_depths, pred_depths, pred_disparities_resized = convert_disps_to_depths_kitti(gt_disparities, pred_depths_org)


    return gt_depths, pred_depths, gt_disparities

def process_depth(gt_depth, pred_depth, gt_disp, split, min_depth, max_depth, garg_crop=True, eigen_crop=False):

        if split == 'eigen':
            mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

            
            if garg_crop or eigen_crop:
                gt_height, gt_width = gt_depth.shape

                # crop used by Garg ECCV16
                # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
                if garg_crop:
                    crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                                     0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)
                # crop we found by trial and error to reproduce Eigen NIPS14 results
                elif eigen_crop:
                    crop = np.array([0.3324324 * gt_height,  0.91351351 * gt_height,   
                                     0.0359477 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
                # Scale matching
                scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
                pred_depth[mask] *= scalor
                pred_depth[pred_depth < min_depth] = min_depth
                pred_depth[pred_depth > max_depth] = max_depth
                gt_depth[gt_depth < min_depth] = min_depth
                gt_depth[gt_depth > max_depth] = max_depth

        if split in ['kitti']:
            mask = gt_disp > 0
            ## median normalize the pred_depth to gt_depth scale
            scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
            pred_depth[mask] *= scalor
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            gt_depth[gt_depth < min_depth] = min_depth
            gt_depth[gt_depth > max_depth] = max_depth

        if split == "make3d":
            mask = gt_disp > 30
            scalor = np.median(gt_depth)/np.median(pred_depth)
            pred_depth *= scalor
            min_depth, max_depth = np.percentile(gt_depth,1), np.percentile(gt_depth, 90)
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            gt_depth[gt_depth < min_depth] = min_depth
            gt_depth[gt_depth > max_depth] = max_depth

        if split == "nyuv2":
            mask = gt_depth > -1
            scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
            pred_depth *= scalor
            pred_depth[pred_depth < min_depth] = min_depth
            pred_depth[pred_depth > max_depth] = max_depth
            gt_depth[gt_depth < min_depth] = min_depth
            gt_depth[gt_depth > max_depth] = max_depth

        return gt_depth, pred_depth, mask

def eval_depth(gt_depths, pred_depths, gt_disparities, split, min_depth=1e-3, max_depth=80):

    num_samples = len(pred_depths)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    for i in range(num_samples):
        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]

        if split == "cs": continue
        gt_depth, pred_depth, mask = process_depth(gt_depth, pred_depth, gt_disparities[i], split, min_depth, max_depth)


        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])
        if abs_rel[i] == 0: continue

        
    if split == "cs": return
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()))

    return abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), a1.mean(), a2.mean(), a3.mean()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Evaluation on the KITTI dataset')
    # parser.add_argument('--vis',                 type=bool,  help='whether to visualize the depth',     default=False)
    # parser.add_argument('--vis_path',            type=str,   help='path to store depth visualization',  default='./depth/')
    parser.add_argument('--split',               type=str,   help='data split, kitti or eigen',         required=True)
    parser.add_argument('--predicted_disp_path', type=str,   help='path to estimated disparities',      required=True)
    parser.add_argument('--gt_path',             type=str,   help='path to ground truth disparities',   required=True)
    parser.add_argument('--min_depth',           type=float, help='minimum depth for evaluation',        default=1e-3)
    parser.add_argument('--max_depth',           type=float, help='maximum depth for evaluation',        default=80)
    parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14',   action='store_true')
    parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16',   action='store_true')

    args = parser.parse_args()

    pred_disparities = np.load(args.predicted_disp_path)
    split, gt_path = args.split, args.gt_path
    min_depth, max_depth = args.min_depth, args.max_depth

    gt_depths, pred_depths, gt_disparities = load_depths(pred_disparities, split, gt_path, test_fn = "")
    eval_depth(gt_depths, pred_depths, gt_disparities, split, min_depth, max_depth)

    
