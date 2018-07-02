## Normal to depth with knowing one depth map and corresponding normal map. 
## Infer the depth map based on neighboring 4/8 pts
## Author: Zhenheng Yang
## Date: 06/07/2017

import tensorflow as tf
import numpy as np
import scipy.misc as sm

def normal2depth_layer(depth_map, normal_map, intrinsic):

    ## depth is of rank 3 [batch, height, width], depth is not inversed
    nei = 2
    depth_map = depth_map[nei:-nei, nei:-nei]
    normal_map = normal_map[nei:-nei, nei:-nei,:]
    depth_dims = depth_map.get_shape().as_list()
    x_coor = tf.range(nei, depth_dims[1]+nei)
    y_coor = tf.range(nei, depth_dims[0]+nei)
    x_ctr, y_ctr = tf.meshgrid(x_coor, y_coor)
    x_ctr = tf.cast(x_ctr, tf.float32)
    y_ctr = tf.cast(y_ctr, tf.float32)
    x0 = x_ctr-nei
    y0 = y_ctr-nei
    x1 = x_ctr+nei
    y1 = y_ctr+nei
    normal_x = normal_map[:,:,0]
    normal_y = normal_map[:,:,1]
    normal_z = normal_map[:,:,2]

    fx = tf.ones(depth_dims) * intrinsic[0]
    fy = tf.ones(depth_dims) * intrinsic[1]
    cx = tf.ones(depth_dims) * intrinsic[2]
    cy = tf.ones(depth_dims) * intrinsic[3]

    ## d_1 = d_0 * ((x_ctr-cx)/fx*normal_x + (y_ctr-cy)/fy*normal_y + normal_z) / ((x_0-cx)/fx*normal_x + (y_0-cy)/fy*normal_y + normal_z)
    numerator = (x_ctr - cx)/fx*normal_x + (y_ctr - cy)/fy*normal_y + normal_z
    denominator_x0 = (x0 - cx)/fx*normal_x + (y_ctr - cy)/fy*normal_y + normal_z + 1e-6
    denominator_y0 = (x_ctr - cx)/fx*normal_x + (y0 - cy)/fy*normal_y + normal_z + 1e-6
    denominator_x1 = (x1 - cx)/fx*normal_x + (y_ctr - cy)/fy*normal_y + normal_z + 1e-6
    denominator_y1 = (x_ctr - cx)/fx*normal_x + (y1 - cy)/fy*normal_y + normal_z + 1e-6
    denominator_x0y0 = (x0 - cx)/fx*normal_x + (y0 - cy)/fy*normal_y + normal_z + 1e-6
    denominator_x0y1 = (x0 - cx)/fx*normal_x + (y1 - cy)/fy*normal_y + normal_z + 1e-6
    denominator_x1y0 = (x1 - cx)/fx*normal_x + (y0 - cy)/fy*normal_y + normal_z + 1e-6
    denominator_x1y1 = (x1 - cx)/fx*normal_x + (y1 - cy)/fy*normal_y + normal_z + 1e-6

    depth_map_x0 = numerator / denominator_x0 * depth_map
    depth_map_y0 = numerator / denominator_y0 * depth_map
    depth_map_x1 = numerator / denominator_y0 * depth_map
    depth_map_y1 = numerator / denominator_y0 * depth_map
    depth_map_x0y0 = numerator / denominator_x0y0 * depth_map
    depth_map_x0y1 = numerator / denominator_x0y1 * depth_map
    depth_map_x1y0 = numerator / denominator_x1y0 * depth_map
    depth_map_x1y1 = numerator / denominator_x1y1 * depth_map

    ## fill the peripheral part (nei) of newly generated with 0
    padding_x0 = [[nei, nei], [0, 2*nei]]
    padding_y0 = [[0, 2*nei], [nei, nei]]
    padding_x1 = [[nei, nei], [2*nei, 0]]
    padding_y1 = [[2*nei, 0], [nei, nei]]
    padding_x0y0 = [[0, 2*nei], [0, 2*nei]]
    padding_x1y0 = [[0, 2*nei], [2*nei, 0]]
    padding_x0y1 = [[2*nei, 0], [0, 2*nei]]
    padding_x1y1 = [[2*nei, 0], [2*nei, 0]]
    
    depth_map_x0 = tf.pad(depth_map_x0-1e6, padding_x0)+1e6
    depth_map_y0 = tf.pad(depth_map_y0-1e6, padding_y0)+1e6
    depth_map_x1 = tf.pad(depth_map_x1-1e6, padding_x1)+1e6
    depth_map_y1 = tf.pad(depth_map_y1-1e6, padding_y1)+1e6
    depth_map_x0y0 = tf.pad(depth_map_x0y0-1e6, padding_x0y0)+1e6
    depth_map_x0y1 = tf.pad(depth_map_x0y1-1e6, padding_x0y1)+1e6
    depth_map_x1y0 = tf.pad(depth_map_x1y0-1e6, padding_x1y0)+1e6
    depth_map_x1y1 = tf.pad(depth_map_x1y1-1e6, padding_x1y1)+1e6

    return depth_map_x0, numerator, denominator_x0

def normal2depth_layer_batch(depth_map, normal_map, intrinsics, tgt_image, nei=1):

    ## depth is of rank 3 [batch, height, width]
    d2n_nei = 3
    depth_map = depth_map[:,d2n_nei+nei:-(d2n_nei+nei), d2n_nei+nei:-(d2n_nei+nei)]
    normal_map = normal_map[:,d2n_nei+nei:-(d2n_nei+nei), d2n_nei+nei:-(d2n_nei+nei), :]

    depth_dims = depth_map.get_shape().as_list()
    x_coor = tf.range(nei, depth_dims[2]+nei)
    y_coor = tf.range(nei, depth_dims[1]+nei)
    x_ctr, y_ctr = tf.meshgrid(x_coor, y_coor)
    x_ctr = tf.cast(x_ctr, tf.float32)
    y_ctr = tf.cast(y_ctr, tf.float32)
    x_ctr_tile = tf.tile(tf.expand_dims(x_ctr, 0), [depth_dims[0], 1, 1])
    y_ctr_tile = tf.tile(tf.expand_dims(y_ctr, 0), [depth_dims[0], 1, 1])
    x0 = x_ctr_tile-nei
    y0 = y_ctr_tile-nei
    x1 = x_ctr_tile+nei
    y1 = y_ctr_tile+nei
    normal_x = normal_map[:,:,:,0]
    normal_y = normal_map[:,:,:,1]
    normal_z = normal_map[:,:,:,2]

    fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
    cx_tile = tf.tile(tf.expand_dims(tf.expand_dims(cx, -1), -1), [1, depth_dims[1], depth_dims[2]])
    cy_tile = tf.tile(tf.expand_dims(tf.expand_dims(cy, -1), -1), [1, depth_dims[1], depth_dims[2]])
    fx_tile = tf.tile(tf.expand_dims(tf.expand_dims(fx, -1), -1), [1, depth_dims[1], depth_dims[2]])
    fy_tile = tf.tile(tf.expand_dims(tf.expand_dims(fy, -1), -1), [1, depth_dims[1], depth_dims[2]])

    numerator = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0 = (x0 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_y0 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1 = (x1 - cx_tile)/fx_tile*normal_x + (y_ctr_tile - cy_tile)/fy_tile*normal_y + normal_z
    denominator_y1 = (x_ctr_tile - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0y0 = (x0 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x0y1 = (x0 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1y0 = (x1 - cx_tile)/fx_tile*normal_x + (y0 - cy_tile)/fy_tile*normal_y + normal_z
    denominator_x1y1 = (x1 - cx_tile)/fx_tile*normal_x + (y1 - cy_tile)/fy_tile*normal_y + normal_z

    mask_x0 = 1e-3 * (tf.cast(tf.equal(denominator_x0, tf.zeros(denominator_x0.get_shape().as_list())), tf.float32))
    denominator_x0 += mask_x0
    mask_y0 = 1e-3 * (tf.cast(tf.equal(denominator_y0, tf.zeros(denominator_y0.get_shape().as_list())), tf.float32))
    denominator_y0 += mask_y0
    mask_x1 = 1e-3 * (tf.cast(tf.equal(denominator_x1, tf.zeros(denominator_x1.get_shape().as_list())), tf.float32))
    denominator_x1 += mask_x1
    mask_y1 = 1e-3 * (tf.cast(tf.equal(denominator_y1, tf.zeros(denominator_y1.get_shape().as_list())), tf.float32))
    denominator_y1 += mask_y1
    mask_x0y0 = 1e-3 * (tf.cast(tf.equal(denominator_x0y0, tf.zeros(denominator_x0y0.get_shape().as_list())), tf.float32))
    denominator_x0y0 += mask_x0y0
    mask_x0y1 = 1e-3 * (tf.cast(tf.equal(denominator_x0y1, tf.zeros(denominator_x0y1.get_shape().as_list())), tf.float32))
    denominator_x0y1 += mask_x0y1
    mask_x1y0 = 1e-3 * (tf.cast(tf.equal(denominator_x1y0, tf.zeros(denominator_x1y0.get_shape().as_list())), tf.float32))
    denominator_x1y0 += mask_x1y0
    mask_x1y1 = 1e-3 * (tf.cast(tf.equal(denominator_x1y1, tf.zeros(denominator_x1y1.get_shape().as_list())), tf.float32))
    denominator_x1y1 += mask_x1y1

    depth_map_x0 = (tf.sigmoid(numerator / denominator_x0 - 1.0) * 2.0 + 4.0) * depth_map
    depth_map_y0 = (tf.sigmoid(numerator / denominator_y0 - 1.0) * 2.0 + 4.0) * depth_map
    depth_map_x1 = (tf.sigmoid(numerator / denominator_x1 - 1.0) * 2.0 + 4.0) * depth_map
    depth_map_y1 = (tf.sigmoid(numerator / denominator_y1 - 1.0) * 2.0 + 4.0) * depth_map

    ## fill the peripheral part (nei) of newly generated with 1e6
    padding_x0 = [[0,0], [d2n_nei+nei, d2n_nei+nei], [d2n_nei, d2n_nei+2*nei]]
    padding_y0 = [[0,0], [d2n_nei, d2n_nei+2*nei], [d2n_nei+nei, d2n_nei+nei]]
    padding_x1 = [[0,0], [d2n_nei+nei, d2n_nei+nei], [d2n_nei+2*nei, d2n_nei]]
    padding_y1 = [[0,0], [d2n_nei+2*nei, d2n_nei], [d2n_nei+nei, d2n_nei+nei]]
    padding_x0y0 = [[0,0], [0, 2*nei], [0, 2*nei]]
    padding_x1y0 = [[0,0], [0, 2*nei], [2*nei, 0]]
    padding_x0y1 = [[0,0], [2*nei, 0], [0, 2*nei]]
    padding_x1y1 = [[0,0], [2*nei, 0], [2*nei, 0]]
    
    depth_map_x0 = tf.pad(depth_map_x0-1e3, padding_x0)+1e3
    depth_map_y0 = tf.pad(depth_map_y0-1e3, padding_y0)+1e3
    depth_map_x1 = tf.pad(depth_map_x1-1e3, padding_x1)+1e3
    depth_map_y1 = tf.pad(depth_map_y1-1e3, padding_y1)+1e3

    img_grad_x0 = tf.pad(tgt_image[:,:,nei:,:] - tgt_image[:,:,:-1*nei,:],[[0,0],[0,0],[0,nei],[0,0]])
    img_grad_y0 = tf.pad(tgt_image[:,nei:,:,:] - tgt_image[:,:-1*nei,:,:],[[0,0],[0,nei],[0,0],[0,0]])
    img_grad_x1 = tf.pad(tgt_image[:,:,2*nei:,:] - tgt_image[:,:,nei:-1*nei,:],[[0,0],[0,0],[2*nei,0],[0,0]])
    img_grad_y1 = tf.pad(tgt_image[:,2*nei:,:,:] - tgt_image[:,nei:-1*nei,:,:],[[0,0],[2*nei,0],[0,0],[0,0]])

    alpha = 0.0
    weights_x0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x0), 3))
    weights_y0 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y0), 3))
    weights_x1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_x1), 3))
    weights_y1 = tf.exp(-1*alpha*tf.reduce_mean(tf.abs(img_grad_y1), 3))
    weights = tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]) / \
                tf.reduce_sum(tf.stack([weights_x0, weights_y0, weights_x1, weights_y1]), 0)

    depth_map_avg = tf.reduce_sum(tf.stack([depth_map_x0, depth_map_y0, depth_map_x1, depth_map_y1])* weights, 0)

    return depth_map_avg

def normalize_l2(vector):
    return tf.nn.l2_normalize(vector, -1)
