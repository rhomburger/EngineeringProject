import tensorflow as tf
import numpy as np
import copy
from sklearn.metrics.pairwise import euclidean_distances #FAST
import time

NUM_LAYERS = 5
PATCH_SIZE = 5#9#7#5#3
PATCH_SHIFT = PATCH_SIZE-1#2



def scan_for_patch(content_patch, style_l):
    min_i=0
    min_j=0
    min_patch = np.sum(np.square(content_patch - style_l[0, min_i:
                        min_i+PATCH_SIZE, min_j: min_j+PATCH_SIZE,:]))

    #TODO limits in both fors: deal with case that the style shape is odd (
    #TODO in that case we dont need to substract PATCH_SIZE
    for i in range(0, style_l.shape[1]-PATCH_SIZE, PATCH_SHIFT):
        for j in range(0, style_l.shape[2]-PATCH_SIZE, PATCH_SHIFT):
            patch_cost = np.sum(np.square(content_patch - style_l[0,
                                                          i:i+PATCH_SIZE,
                                                          j:j+PATCH_SIZE,:]))
            if patch_cost < min_patch:
                min_patch = patch_cost
                min_i = i
                min_j = j

    return style_l[0, min_i : min_i+PATCH_SIZE, min_j: min_j+PATCH_SIZE,:]




def synthesize(content, style):
    """

    :param content: VGG representation of content image
    :param style: VGG representation of style image
    :return:
    """

    weights_3 = np.array([0.25, 0.5, 0.25, 0.5,1,0.5, 0.25,0.5,0.25])
    weights_5 = np.array([0.25, 0.5, 0.5, 0.5, 0.25,
                          0.5, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 0.5,
                          0.25, 0.5, 0.5, 0.5, 0.25])
    weights_7 = np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
    weights_9 = np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5,
                          0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
    weights = weights_5

    syn_img = np.zeros(3, dtype=object)
    syn_img[0] = np.zeros(5, dtype=object)
    syn_img[1] = copy.deepcopy(content[1])
    syn_img[2] = copy.deepcopy(content[2])
        #np.zeros(content.shape)

    for layer in range(NUM_LAYERS):
        print("layer" + str(layer))
        t = time.time()
        syn_img[0][layer] = np.zeros(content[0][layer].shape)
        layer_weights = np.array([weights]*syn_img[0][layer].shape[3]).T.\
                    reshape([PATCH_SIZE, PATCH_SIZE, syn_img[0][layer].shape[3]])
        print(time.time() - t)
        # TODO limits in both fors: deal with case that the style shape is odd (
        # TODO in that case we dont need to substract PATCH_SIZE
        t = time.time()
        for i in range(0, content[0][layer].shape[1]-PATCH_SIZE, PATCH_SHIFT):
            for j in range(0, content[0][layer].shape[2]-PATCH_SIZE, PATCH_SHIFT):
                syn_img[0][layer][0, i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += \
                    layer_weights\
                    *scan_for_patch(content[0][layer][0, i:i+PATCH_SIZE,
                                   j:j+PATCH_SIZE,:], style[0][layer])
        print(time.time() - t)
                #TODO edge cases of average on pixels that are in both patches

    return syn_img


###FAST###

def get_patches(im):
    N = (int)(im.shape[2] - PATCH_SIZE / PATCH_SHIFT)
    M = (int)(im.shape[1] - PATCH_SIZE / PATCH_SHIFT)
    patches = np.zeros([N*M, PATCH_SIZE, PATCH_SIZE, im.shape[3]])
    for i in range(0, im.shape[1] - PATCH_SIZE, PATCH_SHIFT):
        for j in range(0, im.shape[2] - PATCH_SIZE, PATCH_SHIFT):
            patches[N*i+j, :, :, :] = im[0, i:i+PATCH_SIZE, j:j+PATCH_SIZE, :]

    return patches, N, M

def get_image(patches, N, M, d, layer_weights):
    #TODO make sure I deal with edge cases
    im = np.zeros([1, PATCH_SHIFT*N+1, PATCH_SHIFT*M+1, d])
    for i in range(N):
        for j in range(M):
            im[0, PATCH_SHIFT*i:PATCH_SHIFT*i+PATCH_SIZE,
            PATCH_SHIFT*j:PATCH_SHIFT*j+PATCH_SIZE, :] += \
                layer_weights * patches[N*i+j]
    return im


def synthesize_fast(content, style):
    weights = np.array([0.25,0.5, 0.25, 0.5,1,0.5, 0.25,0.5,0.25])

    syn_img = np.zeros(3, dtype=object)
    syn_img[0] = np.zeros(5, dtype=object)
    syn_img[1] = copy.deepcopy(content[1])
    syn_img[2] = copy.deepcopy(content[2])

    for layer in range(NUM_LAYERS):
        print("layer ")
        print(layer)
        syn_img[0][layer] = np.zeros(content[0][layer].shape)
        layer_weights = np.array([weights] * syn_img[0][layer].shape[3]).T. \
            reshape([PATCH_SIZE, PATCH_SIZE, syn_img[0][layer].shape[3]])
        d = content[0][layer].shape[3]

        content_patches, N, M = get_patches(content[0][layer])
        style_patches, _, _ = get_patches(style[0][layer])

        #print(content_patches[0].shape)

        #patches_distances = euclidean_distances(content_patches,
        # style_patches)

        #euclidean distances:
        c = np.array([content_patches]*style_patches.shape[0])
        s = np.array([style_patches] * content_patches.shape[0])

        #print(c.shape)

        patches_distances = np.linalg.norm(c-np.transpose(s,axes=(1,0,2,3,4)),
                                                          axis=4)
        patches_distances = np.linalg.norm(patches_distances, axis=(2,3))



        #print(patches_distances.shape)
        min_distances = np.argmin(patches_distances, axis=1)
        #print(min_distances.shape)
        syn_patches = style_patches[min_distances]
        #print(syn_patches.shape)
        #print(style_patches.shape)


        syn_img[0][layer] = get_image(syn_patches, N, M, d, layer_weights)

    return syn_img


