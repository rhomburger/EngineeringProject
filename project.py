import tensorflow as tf
import numpy as np
import Restoration as r
import utils
import patches
import matplotlib.pyplot as plt

VGG_MEAN = [103.939, 116.779, 123.68]

#Path of Content image and Style image
content_image = "ch/bridge.jpg"
style_image = "ch/scream.jpg"

#Get VGG responses for content and style
content_res = utils.make_features(content_image)
style_res = utils.make_features(style_image)

#Find similar patches
syn_img = patches.synthesize(content_res, style_res)

#Get the new content image in the Restoration Network
res = r.Restoration(im_path="/cs/labs/raananf/rhomburger/code/Restoration",
        model_path ="/cs/labs/raananf/rhomburger/code/Restoration"
                    "/model_net_fix/model_4" ,
        num_epochs=1,
        batch_size=10)
out = res.restore(syn_img, debug=True)

im_restored_bgr = out[0,:,:,:]
im_restored_rgb = np.zeros(im_restored_bgr.shape)
im_restored_rgb[:,:,0] = im_restored_bgr[:,:,2] + VGG_MEAN[2]
im_restored_rgb[:,:,1] = im_restored_bgr[:,:,1] + VGG_MEAN[1]
im_restored_rgb[:,:,2] = im_restored_bgr[:,:,0] + VGG_MEAN[0]

im_restored_rgb = im_restored_rgb/255
im_restored_rgb = np.clip(im_restored_rgb, 0, 1)

plt.imshow(im_restored_rgb)
plt.show()