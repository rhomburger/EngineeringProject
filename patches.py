import tensorflow as tf
import numpy as np
import copy

NUM_LAYERS = 5
PATCH_SIZE = 3
PATCH_SHIFT = 2



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

    weights = np.array([0.25,0.5, 0.25, 0.5,1,0.5, 0.25,0.5,0.25])

    syn_img = np.zeros(3, dtype=object)
    syn_img[0] = np.zeros(5, dtype=object)
    syn_img[1] = copy.deepcopy(content[1])
    syn_img[2] = copy.deepcopy(content[2])
        #np.zeros(content.shape)

    for layer in range(NUM_LAYERS):
        syn_img[0][layer] = np.zeros(content[0][layer].shape)
        layer_weights = np.array([weights]*syn_img[0][layer].shape[3]).T.\
                    reshape([PATCH_SIZE, PATCH_SIZE, syn_img[0][layer].shape[3]])
        # TODO limits in both fors: deal with case that the style shape is odd (
        # TODO in that case we dont need to substract PATCH_SIZE
        for i in range(0, content[0][layer].shape[1]-PATCH_SIZE, PATCH_SHIFT):
            for j in range(0, content[0][layer].shape[2]-PATCH_SIZE, PATCH_SHIFT):
                syn_img[0][layer][0, i:i+PATCH_SIZE, j:j+PATCH_SIZE, :] += \
                    layer_weights\
                    *scan_for_patch(content[0][layer][0, i:i+PATCH_SIZE,
                                   j:j+PATCH_SIZE,:], style[0][layer])

                #TODO edge cases of average on pixels that are in both patches

    return syn_img