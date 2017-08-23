from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import glob
import tensorflow as tf
from tensorflow.python.platform import gfile
import os.path
import re
import hashlib
FLAGS = tf.app.flags.FLAGS

 # 输入和输出文件标志
tf.app.flags.DEFINE_string('image_dir', '',"""\
                           图片文件夹的路径。""")
tf.app.flags.DEFINE_integer('testing_percentage', 10,
                   """图片用于测试的百分比""")
tf.app.flags.DEFINE_integer('validation_percentage', 10,
          """图片用于检定的百分比""")


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    print('当前文件路径：'+FLAGS.image_dir)
    result = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG','png']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            print('子目录与父目录存在一样文件名,文件名称为：'+dir_name)
            continue
        print("正在文件夹'" + dir_name + "'中寻找图片")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('子文件夹中没有找到图片')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
            percentage_hash = (int(hash_name_hashed, 16) % (65536)) * (100 / 65535.0)
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }  
    return result


image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,FLAGS.validation_percentage)

print (image_lists)