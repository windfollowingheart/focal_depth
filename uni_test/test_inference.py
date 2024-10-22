import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# from core.evaluator_interactive import construct_graph, Evaluator, ImageWrapper
from core import evaluator_interactive as evaluator_interactive
import tensorflow as tf
import cv2
import numpy as np
from google.protobuf import text_format
from proto.eval_config_pb2 import EvalConfig

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
flags.DEFINE_string('config_file', '../model/eval.config',
                    'Path of config file')
flags.DEFINE_integer('id', 0,
                    'Path of config file')
FLAGS = flags.FLAGS
id = FLAGS.id
# os.environ["CUDA_VISIBLE_DEVICES"] = '%d'%(id%4)
def get_configs():
    eval_config = EvalConfig()
    with open(FLAGS.config_file, 'r') as f:
        text_format.Merge(f.read(), eval_config)
    tf.logging.info(eval_config)
    return eval_config

def test_one_image(image_path:str, evaluate:evaluator_interactive.Evaluator, depth_split_num:int, save_dir=None):
    im = cv2.imread(image_path)
    im = im.astype(np.float32) / 255
    im = evaluator_interactive.ImageWrapper(im)
    pre_depth = evaluate.run_depth_func(im.im_320)
    depth = pre_depth[0,:,:,0]
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    is_save = True if save_dir else False
    for i in range(depth_split_num):
        depth_percent = i / (depth_split_num - 1)
        focal_depth = depth_min + depth_percent * (depth_max - depth_min)
        im.focal_depth = focal_depth
        save_path=os.path.join(save_dir, os.path.basename(image_path).split(".")[0], \
                                os.path.basename(image_path).split(".")[0] + '_' + str(i) + '.png')
        evaluate.render_by_depth(im, is_save=is_save, save_path=save_path)

def main():
    eval_config = get_configs()
    eval_config.ete_ckpt = "../model/model_end2end/model"
    print("[DEBUG]: start************************************************")
    image_path_arr = [
        r"D:\committers-2022-06\pythonworkplace\AI2\dep_1\data\0_0_8.bmp"
    ]
    save_dir = r"D:\committers-2022-06\pythonworkplace\AI2\dep_1\data\render_result"
    depth_split_num = 10
    is_ok = evaluator_interactive.inference(eval_config, image_path_arr=image_path_arr, \
                                            save_dir=save_dir, depth_split_num=depth_split_num)
    print(is_ok)
    

if __name__ == '__main__':
    main()
        