import Restoration as r
import utils as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.misc



# for i in range(10):
#     name = "x" + str(i) + ".npy"
#     ar = np.load(name)
#     ar[0][4] = np.zeros(ar[0][4].shape)
#     np.save("z_" + name, ar)
#
# exit()


# u.make_examples("/cs/labs/raananf/rhomburger/code/Restoration"
#                 "/training_images", 10)
# exit()

# #creating the features array
# u.make_examples("/cs/labs/raananf/rhomburger/code/Restoration"
#                 "/training_images", 10)

#
#
#
#for training

# res = r.Restoration(
#     im_path="/cs/labs/raananf/rhomburger/code/Restoration",
#     model_path ="/cs/labs/raananf/rhomburger/code/Restoration"
#                 "/model_layers_check"
#                 "/model_new_net_cost" ,
#     num_epochs=200,
#     batch_size=10, reg_factor=500000)
#
# res.train_net()
# exit()
# #



#n2
#n4
#n5 - darker
#n6 light darker


# #
# #
# # #for testing
VGG_MEAN = [103.939, 116.779, 123.68]
res = r.Restoration(
    im_path="/cs/labs/raananf/rhomburger/code/Restoration",
    model_path ="/cs/labs/raananf/rhomburger/code/Restoration/model_layers_check"
                "/model_new_net_cost" ,
    num_epochs=1,
    batch_size=10)


x = np.load("tnx0.npy") #should be tx2

original_im_bgr = np.load("tny0.npy")[0,:,:,:]
original_im_rgb = np.zeros(original_im_bgr.shape)
original_im_rgb[:,:,0] = original_im_bgr[:,:,2] + VGG_MEAN[2]
original_im_rgb[:,:,1] = original_im_bgr[:,:,1] + VGG_MEAN[1]
original_im_rgb[:,:,2] = original_im_bgr[:,:,0] + VGG_MEAN[0]
original_im_rgb = original_im_rgb/255
print(np.max(original_im_rgb))
print(np.min(original_im_rgb))
print(np.max(original_im_bgr))
print(np.min(original_im_bgr))
y = res.restore(x)
np.save("y_restored.npy", y)



im_restored_bgr = y[0,:,:,:]
im_restored_rgb = np.zeros(im_restored_bgr.shape)
im_restored_rgb[:,:,0] = im_restored_bgr[:,:,2] + VGG_MEAN[2]
im_restored_rgb[:,:,1] = im_restored_bgr[:,:,1] + VGG_MEAN[1]
im_restored_rgb[:,:,2] = im_restored_bgr[:,:,0] + VGG_MEAN[0]
im_restored_rgb = im_restored_rgb/255
print(np.max(im_restored_rgb))
print(np.min(im_restored_rgb))
print(np.max(im_restored_bgr))
print(np.min(im_restored_bgr))
#im3 = np.clip(im3, 0, 1)
#scipy.misc.toimage(im).save('testtt.jpg')
im_restored_bgr = im_restored_bgr/255
im_restored_bgr = np.clip(im_restored_bgr, 0, 1)
#matplotlib.image.imsave('testtt.png', im)


#print(np.mean(np.square(im2-im)))

fig = plt.figure()
a=fig.add_subplot(1,2,1)
plt.imshow(original_im_rgb)
a.set_title('Original image')
a=fig.add_subplot(1,2,2)
plt.imshow(im_restored_rgb)
a.set_title('Image after net')
plt.show()



# from skimage import transform as t
# import numpy as np
# from scipy.misc import imread as imread
# import matplotlib.pyplot as plt
#
# im = imread("/cs/labs/raananf/rhomburger/code/Restoration/training_images"
#             "/2007_000241.jpg")
# print(im.shape)
#
# max_dim = max(im.shape[0], im.shape[1])
#
#
# im_resize = t.rescale(im, 224/max_dim)
# print(im_resize.shape)
# plt.imshow(im_resize)
# plt.show()



# checking for similar filters:
# VGG_MEAN = [103.939, 116.779, 123.68]
# res = r.Restoration(
#     im_path="/cs/labs/raananf/rhomburger/code/Restoration",
#     model_path ="/cs/labs/raananf/rhomburger/code/Restoration"
#                 "/model_cost_similar_filters_2"
#                 "/model" ,
#     num_epochs=1,
#     batch_size=10)
#
# x = np.load("tx1.npy")
# x[0][4] *= 1.1
# original_im = np.load("ty1.npy")[0,:,:,:]
# im2 = np.zeros(original_im.shape)
# im2[:,:,0] = original_im[:,:,2] + VGG_MEAN[2]
# im2[:,:,1] = original_im[:,:,1] + VGG_MEAN[1]
# im2[:,:,2] = original_im[:,:,0] + VGG_MEAN[0]
# im2 = im2/255
# print(im2[0:4,0:4,0])
#
# y = res.restore(x)
# im = y[0,:,:,:]
# im3 = np.zeros(im.shape)
# im3[:,:,0] = im[:,:,2] + VGG_MEAN[2]
# im3[:,:,1] = im[:,:,1] + VGG_MEAN[1]
# im3[:,:,2] = im[:,:,0] + VGG_MEAN[0]
# im3 = im3/255
# im3 = np.clip(im3, 0, 1)
# #scipy.misc.toimage(im).save('testtt.jpg')
# im = im/255
# im = np.clip(im, 0, 1)
#
# np.save("sim_fil_t1_conv5.npy", im)

# matplotlib.image.imsave('testtt.png', im)
#
#
# print(np.mean(np.square(im2-im)))
#
# fig = plt.figure()
# a=fig.add_subplot(1,2,1)
# plt.imshow(im2)
# a.set_title('Original image')
# a=fig.add_subplot(1,2,2)
# plt.imshow(im)
# a.set_title('Image after net')
# plt.show()