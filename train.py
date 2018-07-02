from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from LEGOLearner import LEGOLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.0, "Weight for smoothness")
flags.DEFINE_float("normal_smooth_weight", 0.0, "Weight for normal map smoothness")
flags.DEFINE_float("img_grad_weight", 0.0, "Weight for image gradient warping")
flags.DEFINE_float("explain_reg_weight", 0.2, "Weight for explanability regularization")
flags.DEFINE_float("edge_as_explain", 0.0, "Weight for edge as explain mask")
flags.DEFINE_float("edge_mask_weight", 1, "Whether or not use edge prediction, 1 for use")
flags.DEFINE_float("ssim_weight", 1, "Weight for using ssim loss in pixel loss")
flags.DEFINE_float("occ_mask", 0, "occlusion mask")
flags.DEFINE_float("depth_consistency", 0, "if set to 1, there is a depth consistency loss")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("eval_freq", 500, "Evaluation every eval_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_string("checkpoint_continue", "", "From which model it continues training")
flags.DEFINE_string("gpu_id", "0", "GPU id used in training")
flags.DEFINE_float("gpu_fraction", 0.4, "GPU memoery fraction required")
flags.DEFINE_string("eval_txt", "evaluation_kitti.txt", "name of txt files to store the evaluation results")
FLAGS = flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu_id

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
        
    lego = LEGOLearner()
    lego.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
