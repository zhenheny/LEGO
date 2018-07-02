import tensorflow as tf
import numpy as np
import time

def depth2normal_layer(depth_map, intrinsics, inverse):

    ## mask is used to filter the background with infinite depth
    mask = tf.greater(depth_map, tf.zeros(depth_map.get_shape().as_list()))

    if inverse:
        mask_clip = 1e-8 * (1.0-tf.cast(mask, tf.float32)) ## Add black pixels (depth = infinite) with delta
        depth_map += mask_clip
        depth_map = 1.0/depth_map ## inverse depth map
    kitti_shape = depth_map.get_shape().as_list()
    pts_3d_map = compute_3dpts(depth_map, intrinsics)
    print ("shape of pts_3d_map:")
    print (pts_3d_map.get_shape().as_list())
    
    nei = 5

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    pix_num = (depth_map.get_shape().as_list()[0]-2*nei) * (depth_map.get_shape().as_list()[1]-2*nei)
    diff_x0 = tf.reshape(diff_x0, [pix_num, 3])
    diff_y0 = tf.reshape(diff_y0, [pix_num, 3])
    diff_x1 = tf.reshape(diff_x1, [pix_num, 3])
    diff_y1 = tf.reshape(diff_y1, [pix_num, 3])
    diff_x0y0 = tf.reshape(diff_x0y0, [pix_num, 3])
    diff_x0y1 = tf.reshape(diff_x0y1, [pix_num, 3])
    diff_x1y0 = tf.reshape(diff_x1y0, [pix_num, 3])
    diff_x1y1 = tf.reshape(diff_x1y1, [pix_num, 3])

    ## calculate normal by cross product of two vectors
    normals0 = normalize_l2(tf.cross(diff_x1, diff_y1))
    normals1 = normalize_l2(tf.cross(diff_x0, diff_y0))
    normals2 = normalize_l2(tf.cross(diff_x0y1, diff_x0y0))
    normals3 = normalize_l2(tf.cross(diff_x1y0, diff_x1y1))
    
    normal_vector = tf.reduce_sum(tf.concat([[normals0], [normals1], [normals2], [normals3]], 0),0)
    normal_vector = normalize_l2(normal_vector)
    normal_map = tf.reshape(tf.squeeze(normal_vector), [kitti_shape[0]-2*nei]+[kitti_shape[1]-2*nei]+[3])
    print (normal_map.get_shape().as_list())

    normal_map *= tf.tile(tf.expand_dims(tf.cast(mask[nei:-nei, nei:-nei], tf.float32), 2), [1,1,3])
    normal_map = tf.pad(normal_map, [[nei, nei], [nei, nei], [0,0]] ,"CONSTANT")

    return normal_map

def depth2normal_layer_batch(depth_map, intrinsics, inverse, nei=3):

    ## depth_map is in rank 3 [batch, h, w], intrinsics are in rank 2 [batch,4]
    ## mask is used to filter the background with infinite depth
    mask = tf.greater(depth_map, tf.zeros(depth_map.get_shape().as_list()))

    if inverse:
        mask_clip = 1e-8 * (1.0-tf.cast(mask, tf.float32)) ## Add black pixels (depth = infinite) with delta
        depth_map += mask_clip
        depth_map = 1.0/depth_map ## inverse depth map
    kitti_shape = depth_map.get_shape().as_list()
    pts_3d_map = compute_3dpts_batch(depth_map, intrinsics)

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:,nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:,nei:-nei, 0:-(2*nei), :]
    pts_3d_map_y0 = pts_3d_map[:,0:-(2*nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:,nei:-nei, 2*nei:, :]
    pts_3d_map_y1 = pts_3d_map[:,2*nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:,0:-(2*nei), 0:-(2*nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:,2*nei:, 0:-(2*nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:,0:-(2*nei), 2*nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:,2*nei:, 2*nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    pix_num = kitti_shape[0] * (kitti_shape[1]-2*nei) * (kitti_shape[2]-2*nei)
    diff_x0 = tf.reshape(diff_x0, [pix_num, 3])
    diff_y0 = tf.reshape(diff_y0, [pix_num, 3])
    diff_x1 = tf.reshape(diff_x1, [pix_num, 3])
    diff_y1 = tf.reshape(diff_y1, [pix_num, 3])
    diff_x0y0 = tf.reshape(diff_x0y0, [pix_num, 3])
    diff_x0y1 = tf.reshape(diff_x0y1, [pix_num, 3])
    diff_x1y0 = tf.reshape(diff_x1y0, [pix_num, 3])
    diff_x1y1 = tf.reshape(diff_x1y1, [pix_num, 3])

    ## calculate normal by cross product of two vectors
    normals0 = normalize_l2(tf.cross(diff_x1, diff_y1)) #* tf.tile(normals0_mask[:, None], [1,3])
    normals1 = normalize_l2(tf.cross(diff_x0, diff_y0)) #* tf.tile(normals1_mask[:, None], [1,3])
    normals2 = normalize_l2(tf.cross(diff_x0y1, diff_x0y0)) #* tf.tile(normals2_mask[:, None], [1,3])
    normals3 = normalize_l2(tf.cross(diff_x1y0, diff_x1y1)) #* tf.tile(normals3_mask[:, None], [1,3])
    
    normal_vector = tf.reduce_sum(tf.concat([[normals0], [normals1], [normals2], [normals3]], 0),0)
    normal_vector = normalize_l2(normals0)
    normal_map = tf.reshape(tf.squeeze(normal_vector), [kitti_shape[0]]+[kitti_shape[1]-2*nei]+[kitti_shape[2]-2*nei]+[3])

    normal_map *= tf.tile(tf.expand_dims(tf.cast(mask[:, nei:-nei, nei:-nei], tf.float32), -1), [1,1,1,3])
    normal_map = tf.pad(normal_map, [[0,0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")

    return normal_map 

def compute_3dpts(pts, intrinsics):

    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    
    pts_3d = tf.zeros(pts.get_shape().as_list()[:2]+[3])
    pts_z = pts
    x = tf.range(0, pts.get_shape().as_list()[1])
    x = tf.cast(x, tf.float32)
    y = tf.range(0, pts.get_shape().as_list()[0])
    y = tf.cast(y, tf.float32)
    pts_x = (tf.meshgrid(x, y)[0] - tf.ones(pts.get_shape().as_list())*cx) / (tf.ones(pts.get_shape().as_list())*fx) * pts
    pts_y = (tf.meshgrid(x, y)[1] - tf.ones(pts.get_shape().as_list())*cy) / (tf.ones(pts.get_shape().as_list())*fy) * pts
    pts_3d = tf.concat([[pts_x], [pts_y], [pts_z]], 0)
    pts_3d = tf.transpose(pts_3d, perm = [1,2,0])

    return pts_3d

def compute_3dpts_batch(pts, intrinsics):
    
    ## pts is the depth map of rank3 [batch, h, w], intrinsics is in [batch, 4]

    fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3] 
    
    pts_shape = pts.get_shape().as_list()
    pts_3d = tf.zeros(pts.get_shape().as_list()[:2]+[3])
    pts_z = pts
    x = tf.range(0, pts.get_shape().as_list()[2])
    x = tf.cast(x, tf.float32)
    y = tf.range(0, pts.get_shape().as_list()[1])
    y = tf.cast(y, tf.float32)
    cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, pts_shape[1], pts_shape[2]])
    fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, pts_shape[1], pts_shape[2]])
    pts_x = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[0], 0), [pts_shape[0], 1, 1]) - cx_tile) / fx_tile * pts
    pts_y = (tf.tile(tf.expand_dims(tf.meshgrid(x, y)[1], 0), [pts_shape[0], 1, 1]) - cy_tile) / fy_tile * pts
    pts_3d = tf.concat([[pts_x], [pts_y], [pts_z]], 0)
    pts_3d = tf.transpose(pts_3d, perm = [1,2,3,0])

    return pts_3d

def normalize_l2(vector):
    return tf.nn.l2_normalize(vector, 1, epsilon=1e-20)