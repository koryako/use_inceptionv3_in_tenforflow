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
import sys
import tarfile
import numpy as np
from six.moves import urllib
FLAGS = tf.app.flags.FLAGS

 # 输入和输出文件标志
tf.app.flags.DEFINE_string('image_dir', '',"""\
                           图片文件夹的路径。""")
tf.app.flags.DEFINE_integer('testing_percentage', 10,
                   """图片用于测试的百分比""")
tf.app.flags.DEFINE_integer('validation_percentage', 10,
          """图片用于检定的百分比""")
tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """classify_image_graph_def.pb，"""
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2_challenge_label_map_proto.pbtxt的路径。""")
tf.app.flags.DEFINE_string('output_graph', '/tmp/output_graph.pb',
                                   """训练图表保存到哪里？""")
tf.app.flags.DEFINE_string('output_labels', '/tmp/output_labels.txt',
                         """训练图表的标签保存到哪里？""")

 # 详细的训练配置
tf.app.flags.DEFINE_integer('how_many_training_steps', 0,
                            """在结束之前，需要训练多少步？""")
tf.app.flags.DEFINE_float('learning_rate', 0.01,
                          """在训练的时候设置多大的学习率？""")
tf.app.flags.DEFINE_integer('testing_percentage', 10,
                   """图片用于测试的百分比""")
tf.app.flags.DEFINE_integer('validation_percentage', 10,
          """图片用于检定的百分比""")
tf.app.flags.DEFINE_integer('eval_step_interval', 10,
                            """评估培训结果的频率？""")
tf.app.flags.DEFINE_integer('train_batch_size', 100,
                            """一次训练多少张照片？""")
tf.app.flags.DEFINE_integer('test_batch_size', 500,
                            """一次测试多少张照片？"""
                            """这个测试集只使用很少来验证"""
                             """模型的整体精度。""")
tf.app.flags.DEFINE_integer(
    'validation_batch_size', 100,
    """有多少图片在一个评估批量使用。这个验证集"""
    """被使用的频率比测试集多, 这也是一个早期的指标"""
    """模型有多精确在训练期间。""")

# 文件系统cache所在目录
tf.app.flags.DEFINE_string('model_dir', '/tmp/imagenet',
                           """classify_image_graph_def.pb，"""
                           """imagenet_synset_to_human_label_map.txt, and """
                           """imagenet_2_challenge_label_map_proto.pbtxt的路径。""")
tf.app.flags.DEFINE_string(
   'bottleneck_dir', '/tmp/bottleneck',
   """cache bottleneck 层作为值的文件集。""")
tf.app.flags.DEFINE_string('final_tensor_name', 'final_result',
                          """分类输出层的名称"""
                         """在重新训练时。""")

 # 控制扭曲参数在训练期间
tf.app.flags.DEFINE_boolean(
     'flip_left_right', False,
    """是否随机对半水平翻转训练图片。""")
tf.app.flags.DEFINE_integer(
    'random_crop', 0,
   """A percentage determining how much of a margin to randomly crop off the"""
    """ training images.""")
tf.app.flags.DEFINE_integer(
    'random_scale', 0,
    """A percentage determining how much to randomly scale up the size of the"""
    """ training images by.""")
tf.app.flags.DEFINE_integer(
     'random_brightness', 0,
    """A percentage determining how much to randomly multiply the training"""
    """ image input pixels up or down by.""")


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 8
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

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

def maybe_download_and_extract(data_url):
  """Download and extract model tar file.
  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  Args:
    data_url: Web location of the tar file containing the pretrained model.
  """
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  
  if not os.path.exists(filepath):
    print(filepath)
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    tf.logging.info('Successfully downloaded', filename, statinfo.st_size,'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


"""
从保存的GraphDef文件创建一个图形，然后返回一个图形对象。
Returns：
包含已经训练过的Inception网络图形，和各种各样的tensors，我们将要控制的。
"""

def create_inception_graph():
    with tf.Session() as sess:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
           graph_def = tf.GraphDef()
           graph_def.ParseFromString(f.read())
           bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
               tf.import_graph_def(graph_def, name='', return_elements=[
                   BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                   RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def should_distort_images(flip_left_right, random_crop, random_scale,
                           random_brightness):
   """是否启用了扭曲,从输入标志。
 
   Args:
     flip_left_right: Boolean是否随机水平反应图片。
     random_crop: 整数百分比设置使用总利润在裁剪盒子。
     random_scale: 改变规模的整数百分比是多少
     random_brightness: 整数范围内随机乘以像素的值。
 
   Returns:
     Boolean值表明，是否有扭曲的值需要被应用。
   """
   return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
           (random_brightness != 0))
 
def add_input_distortions(flip_left_right, random_crop, random_scale,
                           random_brightness):
   """创建操作来应用规定的扭曲。

  在训练的时候，可以帮助改善结果，如果我们运行图片通过简单扭曲，例如修剪，缩放，翻转。
这些反映这种变化是我们在现实世界所期待的，所以可以帮助训练模型，更高效地应对自然的数据。
这里我们提供参数和构造一个操作网络在一张照片应用他们。

  Cropping
  ~~~~~~~~

  Cropping is done by placing a bounding box at a random position in the full
  image. The cropping parameter controls the size of that box relative to the
  input image. If it's zero, then the box is the same size as the input and no
  cropping is performed. If the value is 50%, then the crop box will be half the
  width and height of the input. In a diagram it looks like this:

  <       width         >
  +---------------------+
  |                     |
  |   width - crop%     |
  |    <      >         |
  |    +------+         |
  |    |      |         |
  |    |      |         |
  |    |      |         |
  |    +------+         |
  |                     |
  |                     |
  +---------------------+

  Scaling
  ~~~~~~~

  Scaling is a lot like cropping, except that the bounding box is always
  centered and its size varies randomly within the given range. For example if
  the scale percentage is zero, then the bounding box is the same size as the
  input and no scaling is applied. If it's 50%, then the bounding box will be in
  a random range between half the width and height and full size.

  Args:
    flip_left_right: Boolean是否随机水平反应图片。
    random_crop: 整数百分比设置使用总利润在裁剪盒子。
    random_scale: 改变规模的整数百分比是多少
    random_brightness: 整数范围内随机乘以像素的值。

  Returns:
    jpeg的输入层和扭曲最后的tensor。
  """
  jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
  decoded_image = tf.image.decode_jpeg(jpeg_data)
  decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
  margin_scale = 1.0 + (random_crop / 100.0)
  resize_scale = 1.0 + (random_scale / 100.0)
  margin_scale_value = tf.constant(margin_scale)
  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=1.0,
                                         maxval=resize_scale)
  scale_value = tf.mul(margin_scale_value, resize_scale_value)
  precrop_width = tf.mul(scale_value, MODEL_INPUT_WIDTH)
  precrop_height = tf.mul(scale_value, MODEL_INPUT_HEIGHT)
  precrop_shape = tf.pack([precrop_height, precrop_width])
  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                              precrop_shape_as_int)
  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
  cropped_image = tf.random_crop(precropped_image_3d,
                                 [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                  MODEL_INPUT_DEPTH])
  if flip_left_right:
    flipped_image = tf.image.random_flip_left_right(cropped_image)
  else:
    flipped_image = cropped_image
  brightness_min = 1.0 - (random_brightness / 100.0)
  brightness_max = 1.0 + (random_brightness / 100.0)
  brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                       minval=brightness_min,
                                       maxval=brightness_max)
  brightened_image = tf.mul(flipped_image, brightness_value)
  distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
  return jpeg_data, distort_result


def main():
    maybe_download_and_extract(DATA_URL)
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('没有找到图片文件夹： ' + FLAGS.image_dir)
    if class_count == 1:
        print('只有一个合法的文件夹 ' + FLAGS.image_dir +' - 不能满足多分类模型.')


# 看看这个命令行标志意味着我们使用任意扭曲。 
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
    sess = tf.Session()
 
    if do_distort_images:
     # 我们将运用扭曲，所以我们需要设置操作。
        distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
           FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
           FLAGS.random_brightness)
    else:
     # 我们将确保我们已经计算了'bottleneck'图像摘要和缓存
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor)
 
   # 添加新的层，我们将训练。 
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                          FLAGS.final_tensor_name,
                                          bottleneck_tensor)
 
   # 将所有的权重设置为初始默认值。 
    init = tf.initialize_all_variables()
    sess.run(init)
 
   # 创建操作，我们需要评估我们的新层的准确性。
    evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

# 运行在命令行上的要求的许多周期的训练。 
   for i in range(FLAGS.how_many_training_steps):
     #获取一个输入bottleneck值，要么计算新的值每一次，或者从存储在硬盘的缓存获得。
     if do_distort_images:
       train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
           sess, image_lists, FLAGS.train_batch_size, 'training',
           FLAGS.image_dir, distorted_jpeg_data_tensor,
          distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
     else:
       train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'training',
           FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
           bottleneck_tensor)
     # Feed的bottlenecks和ground truth 到图表，并且运行一个训练步骤。
     sess.run(train_step,
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
     # 每一个如此，打印出有多么好的图形训练。 
     is_last_step = (i + 1 == FLAGS.how_many_training_steps)
     if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
       train_accuracy, cross_entropy_value = sess.run(
           [evaluation_step, cross_entropy],
           feed_dict={bottleneck_input: train_bottlenecks,
                      ground_truth_input: train_ground_truth})
       print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                       train_accuracy * 100))
       print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                  cross_entropy_value))
       validation_bottlenecks, validation_ground_truth = (
           get_random_cached_bottlenecks(
               sess, image_lists, FLAGS.validation_batch_size, 'validation',
               FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
               bottleneck_tensor))
       validation_accuracy = sess.run(
           evaluation_step,
           feed_dict={bottleneck_input: validation_bottlenecks,
                      ground_truth_input: validation_ground_truth})
      print('%s: Step %d: Validation accuracy = %.1f%%' %
             (datetime.now(), i, validation_accuracy * 100))
 
   # 我们已经完成了所有的训练，所以在一些新的测试中运行了最后的测试评估 
   test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
       sess, image_lists, FLAGS.test_batch_size, 'testing',
       FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
       bottleneck_tensor)
   test_accuracy = sess.run(
      evaluation_step,
      feed_dict={bottleneck_input: test_bottlenecks,
                  ground_truth_input: test_ground_truth})
   print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
 
   # 把训练的图表和标签与存储为常量的权重。 
   output_graph_def = graph_util.convert_variables_to_constants(
       sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
   with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
     f.write(output_graph_def.SerializeToString())
   with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
     f.write('\n'.join(image_lists.keys()) + '\n')

if __name__ == '__main__':
    main()