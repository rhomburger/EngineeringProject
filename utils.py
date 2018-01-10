import tensorflow as tf
import numpy as np
import vgg19
import os
import random
from scipy.misc import imread as imread
import skimage.transform as tr
from PIL import Image

VGG_MEAN = [103.939, 116.779, 123.68]
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


#todo: create a different graph to run vgg19. utils should return numpy arrays!


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


def get_im(path, full_net):
    filenames = list_images(path, True)
    i=0
    while True:
        file = filenames[i]
        im = imread(file)
        im = tr.resize(im, (224, 224))

        im_vgg= vgg19.Vgg19()
        x = get_features(im, im_vgg, full_net)
        y = np.zeros([1, im.shape[0], im.shape[1], im.shape[2]])
        y[0,:,:,:] = im

        i= i+1

        yield (x,y)

def get_example(path, batch_size):
    i=0
    while True:

        x = np.load("nx" + str(i) + ".npy")
        y = np.load("ny" + str(i) + ".npy").astype(np.float32)

        i=((i+1) % batch_size)

        yield (x,y)


# def get_batch(path, batch_size, full_net):
#     """
#     Function get a list of images, and returns a python generator of couples of arrays - array of images and array of
#     corrupted images.
#     :param filenames: list of images locations
#     :param batch_size: amount of images to return
#     :param corruption_func: function that corrupts the original image
#     :param crop_size: size of path
#     :return: python generator of tuples, each element is an array of batch_size with images
#     """
#
#     filenames = list_images(path, True)
#
#     cache = {}
#     while True:
#         batch_x = np.zeros(shape=batch_size, dtype=np.object)
#         batch_y = np.zeros(shape=batch_size, dtype=np.object)
#
#         for i in range(batch_size):
#             file = filenames[random.randint(0, len(filenames) - 1)]
#             if file in cache:
#                 im = cache[file]
#             else:
#                 im = imread(file)
#                 cache[file] = im
#
#             im_vgg = vgg19.Vgg19()
#             features = get_features(im)
#             batch_x[i] = features
#             batch_y[i] = im
#
#         yield (batch_x, batch_y)


def make_examples(path, num_examples):
    filenames = list_images(path, True)
    for i in range(num_examples):
        file = filenames[i]
        im = Image.open(file)#imread(file)
        # im = im.astype(np.float32)
        # im = im/255
        im = im.resize((224,224), Image.ANTIALIAS)#tr.resize(im, (224, 224))
        im = np.array(im)
        print(np.max(im))
        print(np.min(im))
        im = im.astype(np.float32)
        #im = im*255.



        im_dim = np.zeros(
            [1, im.shape[0], im.shape[1], im.shape[2]]).astype(np.float32)
        im_dim[0, :, :, 0] = im[:, :, 2] - VGG_MEAN[0]
        im_dim[0, :, :, 1] = im[:, :, 1] - VGG_MEAN[1]
        im_dim[0, :, :, 2] = im[:, :, 0] - VGG_MEAN[2]
        im_dim /= 255

        #im_dim = im_dim.astype(np.float32)



        # im_dim2 = np.zeros(
        #     [1, im.shape[0], im.shape[1], im.shape[2]]).astype(np.float32)
        # im_dim2[:, :, :, 0] = im[:, :, 0]
        # im_dim2[:, :, :, 1] = im[:, :, 1]
        # im_dim2[:, :, :, 2] = im[:, :, 2]

        #im_dim2 = im_dim.astype(np.float32)


        # print(np.min(im))
        # print(im.dtype)
        # print(np.min(im_dim))
        # print(im_dim.dtype)
        # print(np.min(im_dim2))
        # print(im_dim2.dtype)
        # x=1/0


        tf.reset_default_graph()
        with tf.Session() as sess:

            # # Convert RGB to BGR
            # rgb_scaled = im_tensor
            # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            # bgr = tf.concat(axis=3, values=[
            #     blue - VGG_MEAN[0],
            #     green - VGG_MEAN[1],
            #     red - VGG_MEAN[2],
            # ])
            # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
            im_tensor = tf.stack(im_dim)
            bgr = im_tensor
            vgg = vgg19.Vgg19()
            vgg.build(bgr)

            # convs = [vgg.conv5_4, vgg.conv4_4, vgg.conv3_4, vgg.conv2_2,
            #                  vgg.conv1_2]
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

            y = im_dim#sess.run(bgr)

            x_name = "norm_x" + str(i)
            y_name = "norm_y" + str(i)
            np.save(x_name, x)
            np.save(y_name, y)


        # im_vgg = vgg19.Vgg19()
        # x = get_features2(im)

        # y = np.zeros([1, im.shape[0], im.shape[1], im.shape[2]])
        # y[0,:,:,:] = im
        # print("type of x:")
        # print(x.dtype)
        # print("type of y:")
        # print(y.dtype)



        # n = [n.name for n in tf.get_default_graph().as_graph_def().node]
        # gname = "graph" + str(i + 1)
        # f = open(gname, 'w')
        # for st in n:
        #     f.write(st)
        #     f.write("\n")
        # f.close()



def get_features2(image):


    tf.reset_default_graph()
    with tf.Session() as sess:
        im_dim = np.zeros([1, image.shape[0], image.shape[1], image.shape[2]])
        im_dim[0, :, :, :] = image
        print(im_dim.dtype)
        print(np.max(im_dim))
        im_dim = im_dim.astype(np.float32)
        print(im_dim.dtype)
        print(np.max(im_dim))
        #im_dim = im_dim / 255
        im_tensor = tf.stack(im_dim)

        vgg = vgg19.Vgg19()
        vgg.build(im_tensor)

        convs = [vgg.conv5_4, vgg.conv4_4, vgg.conv3_4, vgg.conv2_2,
                         vgg.conv1_2]
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


def get_features(image, vgg, full_net):

    #convert the image to tensor

    im_dim = np.zeros([1,image.shape[0], image.shape[1], image.shape[2]])
    im_dim[0,:,:,:] = image
    im_dim=im_dim.astype(np.float32)
    im_tensor = tf.stack(im_dim)
        #todo The images is a tensor with shape [None, 224, 224, 3].

    #feed the vgg net in the image
    vgg.build(im_tensor)

    #build features vector
    # if full_net:
    #     convs = [[vgg.conv5_4, vgg.conv5_3, vgg.conv5_2, vgg.conv5_1],
    #            [vgg.conv4_4, vgg.conv4_3, vgg.conv4_2, vgg.conv4_1],
    #         [vgg.conv3_4, vgg.conv3_3, vgg.conv3_2, vgg.conv3_1],
    #         [vgg.conv2_2, vgg.conv2_1],
    #         [vgg.conv1_2, vgg.conv1_1]]
    # else:
    #     convs = [[vgg.conv5_4],
    #            [vgg.conv4_4],
    #         [vgg.conv3_4],
    #         [vgg.conv2_2],
    #         [vgg.conv1_2]]
    pools = [vgg.pool4, vgg.pool3, vgg.pool2, vgg.pool1]

    features = []

    features.append([vgg.conv5_4, vgg.conv4_4, vgg.conv3_4, vgg.conv2_2,
                      vgg.conv1_2])

    for i in range(2):
        features.append([])
    dims = []
    ind = []

    for l in range(4):
        # for c in convs:
        #     features[0].append(c)
        dims.append([pools[l][0].shape[0], pools[l][0].shape[0]]) #todo
        #  verify 0,1 are the correct indexes
        ind.append(pools[l][1])
    # for c in convs[4]:
    #     features[0].append(c)

    features[1] = dims
    features[2] = ind

    return features


# def get_features(net, path, num_images):
#     vgg_weights, vgg_mean_pixel = vgg.load_net(net)
#     pooling = tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
#                 padding='SAME') #max pooling
#
#     dataset = np.array()
#
#     for ___:
#         im = load_image(im)
#         shape = (1,) + im.shape
#         features = {}
#
#         # compute content features in feedforward mode
#         g = tf.Graph()
#         with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
#             image = tf.placeholder('float', shape=shape)
#             net = vgg.net_preloaded(vgg_weights, image, pooling)
#             content_pre = np.array([vgg.preprocess(im, vgg_mean_pixel)])
#             for layer in CONTENT_LAYERS:
#                 features[layer] = net[layer].eval(feed_dict={image: content_pre})
#
#             dataset = np.append(dataset, (features, im))
#
#     return dataset
