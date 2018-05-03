import tensorflow as tf
import numpy as np
import vgg19
import os
import random
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def list_images(path, use_shuffle=True):
    """Returns a list of paths to images found at the specified directory.

    Arguments:
    path -- path to a directory to search for images.
    use_shuffle -- option to shuffle order of files. Uses a fixed shuffled order.
    """
    def is_image(filename):
        return os.path.splitext(filename)[-1][1:].lower() in ['jpg', 'png']
    images = list(map(lambda x: os.path.join(path, x), filter(is_image, os.listdir(path))))
    # Shuffle with a fixed seed without affecting global state
    if use_shuffle:
        s = random.getstate()
        random.seed(1234)
        random.shuffle(images)
        random.setstate(s)
    return images

def get_example(path, batch_size):
    """
    Generator of images for the restoration net training
    """
    i=0
    while True:

        x = np.load("test_x" + str(i) + ".npy")
        y = np.load("test_y" + str(i) + ".npy").astype(np.float32)

        i=((i+1) % batch_size)

        yield (x,y)

def make_features(im_path):
    """
    Return image VGG's features
    """
    im = Image.open(im_path)
    im = im.resize(((int)(im.size[0]/1),(int)(im.size[1]/1)),
                   Image.ANTIALIAS)
    im = np.array(im)
    im = im.astype(np.float32)

    im_dim = np.zeros(
        [1, im.shape[0], im.shape[1], im.shape[2]]).astype(np.float32)
    im_dim[0, :, :, 0] = im[:, :, 2] - VGG_MEAN[0]
    im_dim[0, :, :, 1] = im[:, :, 1] - VGG_MEAN[1]
    im_dim[0, :, :, 2] = im[:, :, 0] - VGG_MEAN[2]

    tf.reset_default_graph()
    with tf.Session() as sess:
        im_tensor = tf.stack(im_dim)
        bgr = im_tensor
        vgg = vgg19.Vgg19()
        vgg.build(bgr)

        convs = [vgg.conv5_1, vgg.conv4_1, vgg.conv3_1, vgg.conv2_1,
                 vgg.conv1_1]
        pools = [vgg.pool4, vgg.pool3, vgg.pool2, vgg.pool1]

        features = np.zeros(3, dtype=object)

        #convolution layers
        features_convs = np.zeros(5, dtype=object)
        for c in range(5):
            features_convs[c] = sess.run(convs[c])
        features[0] = features_convs

        #pooling layers
        features_dims = np.zeros(4, dtype=object)
        features_ind = np.zeros(4, dtype=object)
        for p in range(4):
            features_dims[p] = [features_convs[p+1].shape[1],
                                features_convs[p+1].shape[2]]
            features_ind[p] = sess.run(pools[p][1])
        features[1] = features_dims
        features[2] = features_ind

        return features


def make_examples(path, num_examples):
    """
    Make examples for training of the restoration net.
    """
    filenames = list_images(path, True)
    for i in range(num_examples):
        file = filenames[i]
        im = Image.open(file)#imread(file)
        im = np.array(im)
        im = im.astype(np.float32)

        im_dim = np.zeros(
            [1, im.shape[0], im.shape[1], im.shape[2]]).astype(np.float32)
        im_dim[0, :, :, 0] = im[:, :, 2] - VGG_MEAN[0]
        im_dim[0, :, :, 1] = im[:, :, 1] - VGG_MEAN[1]
        im_dim[0, :, :, 2] = im[:, :, 0] - VGG_MEAN[2]

        tf.reset_default_graph()
        with tf.Session() as sess:
            im_tensor = tf.stack(im_dim)
            bgr = im_tensor
            vgg = vgg19.Vgg19()
            vgg.build(bgr)

            convs = [vgg.conv5_1, vgg.conv4_1, vgg.conv3_1, vgg.conv2_1,
                     vgg.conv1_1]
            pools = [vgg.pool4, vgg.pool3, vgg.pool2, vgg.pool1]

            features = np.zeros(3, dtype=object)

            #convolution layers
            features_convs = np.zeros(5, dtype=object)
            for c in range(5):
                features_convs[c] = sess.run(convs[c])
            features[0] = features_convs

            #pooling layers
            features_dims = np.zeros(4, dtype=object)
            features_ind = np.zeros(4, dtype=object)
            for p in range(4):
                features_dims[p] = [features_convs[p+1].shape[1],
                                    features_convs[p+1].shape[2]]
                features_ind[p] = sess.run(pools[p][1])
            features[1] = features_dims
            features[2] = features_ind

            x = features
            y = im_dim

            x_name = "test_x" + str(i)
            y_name = "test_y" + str(i)
            np.save(x_name, x)
            np.save(y_name, y)
