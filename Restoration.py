import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.misc
import matplotlib.image




VGG_MEAN = [103.939, 116.779, 123.68]



class Restoration:


	def __init__(self, im_path, model_path, num_epochs, batch_size,f_size=3,
				 full_net=False, reg_factor=1, learning_rate = 0.001):
		# maybe here I should reset the graph?

		self.path = im_path
		self.model_location = model_path
		self.filter_size = f_size # size of deconvolution filter

		# self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
         #    pred, y))
        #
		# self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)


		self.weights = dict() # dictionary of weights for each deconvolution layer
		self.biases = dict() # dictionary of biases for each deconvolution layer

		self.convs = []
		self.inputs = dict() # tensorflow's placeholders for each layer (as each layer also gets an input)
		self.pool_dims = [] # placeholders for original pre-pooling dims
		self.pool_ind = [] # placeholders for pooling argmax

		self.layers = []
		
		self.alpha = reg_factor #regularization factor of the loss function
		self.batch_size = batch_size		
		self.num_batches = len(utils.list_images(self.path)) / self.batch_size
		self.num_epochs = num_epochs
		self.num_images = batch_size
		self.means = []#np.array([], dtype=np.float32)
		self.full_net = full_net #true if we want all responses to connect to the layers, false if only one response per block

		self.random_weights = 0.05

		self.random_biases = 0
		self.learning_rate = learning_rate

		self.beta = 0.05
		self.means_weights = []
		self.fourier = []
		self.gradients = []


	def deconv(self, in_block, weights, name):

		return tf.nn.conv2d(input=in_block, filter=weights,
								strides=[1,1,1,1], padding="SAME",
							name=name)


	def unpool2d(self, p, ind, org_dims, ksize=[1, 2, 2, 1]):
		# UnPool
		#todo my addition:
		ind = tf.stack(ind)
		#print("ind: ",ind)

		dims = tf.shape(p, out_type=tf.int32)
		flt = tf.reshape(tf.eye(dims[3]), [1, 1, dims[3], dims[3]])

		mask = tf.one_hot(ind, 4)

		m1 = tf.reshape(tf.slice(mask, [0, 0, 0, 0, 0], [-1, -1, -1, -1, 1]), dims)
		m2 = tf.reshape(tf.slice(mask, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 1]), dims)
		m3 = tf.reshape(tf.slice(mask, [0, 0, 0, 0, 2], [-1, -1, -1, -1, 1]), dims)
		m4 = tf.reshape(tf.slice(mask, [0, 0, 0, 0, 3], [-1, -1, -1, -1, 1]), dims)

		u = tf.slice(tf.nn.conv2d_transpose(tf.multiply(m1, p), flt, tf.cast(
            [dims[0], 2 * dims[1], 2 * dims[2], dims[3]], tf.int32), [1, 2, 2, 1]),
                     [0, 0, 0, 0], [-1, org_dims[0], org_dims[1], -1])
		u = tf.add(u, tf.slice(tf.pad(
            tf.nn.conv2d_transpose(tf.multiply(m2, p), flt, tf.cast(
                [dims[0], 2 * dims[1], 2 * dims[2], dims[3]], tf.int32),
                                   [1, 2, 2, 1]),
            [[0, 0], [1, 0], [0, 0], [0, 0]]), [0, 0, 0, 0],
                               [-1, org_dims[0], org_dims[1], -1]))
		u = tf.add(u, tf.slice(tf.pad(
            tf.nn.conv2d_transpose(tf.multiply(m3, p), flt, tf.cast(
                [dims[0], 2 * dims[1], 2 * dims[2], dims[3]], tf.int32),
                                   [1, 2, 2, 1]),
            [[0, 0], [0, 0], [1, 0], [0, 0]]), [0, 0, 0, 0],
                               [-1, org_dims[0], org_dims[1], -1]))
		u = tf.add(u, tf.slice(tf.pad(
            tf.nn.conv2d_transpose(tf.multiply(m4, p), flt, tf.cast(
                [dims[0], 2 * dims[1], 2 * dims[2], dims[3]], tf.int32),
                                   [1, 2, 2, 1]),
            [[0, 0], [1, 0], [1, 0], [0, 0]]), [0, 0, 0, 0],
                               [-1, org_dims[0], org_dims[1], -1]))

		return u

	def crop_and_concat(self, x1,x2):
		x1_shape = tf.shape(x1)
		x2_shape = tf.shape(x2)
		# offsets for the top left corner of the crop
		offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
		size = [-1, x2_shape[1], x2_shape[2], -1]
		x1_crop = tf.slice(x1, offsets, size)
		return tf.concat([x1_crop, x2], 3)




		
	def layer_to_name(self, i, j):
		return "deconv" + str(i) + "_" + str(j)
	
	def name_to_layer(self, name):
		return (name[6], name[8])
		
		


	# def unpooling(self, res, stam=-2):
	#
	# 	dims = tf.placeholder("int32", shape=[2])
	# 	self.pool_dims.append(dims)
	# 	ind = tf.placeholder("int64", shape=[None, None, None, None])
	# 	self.pool_ind.append(ind)
    #
	# 	p = self.unpool2d(res, ind, dims)
	# 	#print("after pool: ", p)
	# 	return p

	def unpooling(self, inp, stam=-1):
		dims = tf.placeholder("int32", shape=[2])
		self.pool_dims.append(dims)
		p = tf.image.resize_nearest_neighbor(inp, size=dims)
		return p





	def build_net2(self):
		self.label = tf.placeholder("float32", shape=[1, None, None, 3])
		self.convs.append(tf.placeholder("float32", shape=[1, None, None,
														   512],
										 name="input1"))
		self.convs.append(tf.placeholder("float32", shape=[1, None, None,
														   512],
										 name="input2"))
		self.convs.append(tf.placeholder("float32", shape=[1, None, None,
														   256],
										 name="input3"))
		self.convs.append(tf.placeholder("float32", shape=[1, None, None,
														   128],
										 name="input4"))
		self.convs.append(tf.placeholder("float32", shape=[1, None, None,
														   64], name="input5"))

		self.weights[4] = tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,512,512]),
									  name="weights1")
		self.biases[4] = tf.Variable(self.random_biases*tf.random_normal([
			512]), name="biases1")
		t = tf.nn.relu(self.deconv(self.convs[0], self.weights[4], "deconv1") +
									   self.biases[4], name="relu1")
		#self.means.append(tf.reduce_mean(t))
		t = self.unpooling(t, 4)


		self.weights[3] = tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*512,256]),
									  name="weights2")
		self.biases[3] = tf.Variable(self.random_biases*tf.random_normal([
			256]), name="biases2")

		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[1]], 3),
								   self.weights[3], "deconv2") +
					   self.biases[3], name="relu2")
		#self.means.append(tf.reduce_mean(t))
		t = self.unpooling(t, 3)


		self.weights[2] = tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*256,128]),
									  name="weights3")
		self.biases[2] = tf.Variable(self.random_biases*tf.random_normal([
			128]), name="biases3")
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[2]], 3),
								   self.weights[2], "deconv3") +
					   self.biases[2], name="relu3")
		#self.means.append(tf.reduce_mean(t))
		t = self.unpooling(t, 2)


		self.weights[1] = tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*128,64]), name="weights4")
		self.biases[1] = tf.Variable(self.random_biases*tf.random_normal([
			64]), name="biases4")
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[3]], 3),
								   self.weights[1], "deconv4") +
					   self.biases[1], name="relu4")
		#self.means.append(tf.reduce_mean(t))
		t = self.unpooling(t, 1)


		self.weights[0] = tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*64,64]),
									  name="weights5")
		self.biases[0] = tf.Variable(self.random_biases*tf.random_normal([
			64]), name="biases5")
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[4]], 3),
								   self.weights[0], "deconv5") +
					   self.biases[0], name="relu5")
		#self.means.append(tf.reduce_mean(t))
		#no unpooling


		self.weights[-1] = tf.Variable(self.random_weights*tf.random_normal([
			self.filter_size, self.filter_size, 64, 3]), name="weights6")
		self.biases[-1] = tf.Variable(self.random_biases*tf.random_normal([
			3]), name="biases6")
		self.out = self.deconv(t, self.weights[-1], "deconv6-out") + \
				   self.biases[-1]

		for i in range(5):
			self.gradients.append(tf.sqrt(tf.reduce_mean(tf.square(
				tf.gradients(self.out, self.convs[i])))))


			#prev chosen:
			# self.gradients.append(tf.reduce_mean(tf.gradients(tf.sqrt(
			# 	tf.reduce_mean(tf.square(
			# 	self.out))), self.convs[i])))

			#OR

			# self.gradients.append(tf.sqrt(tf.reduce_mean(tf.square(
			# 	tf.gradients(tf.sqrt(tf.reduce_mean(tf.square(
			# 	self.out))),self.convs[i])))))


		return self.out



	def train_net(self):
		"""
        trains the input net
        :param net:
        :param examples_location:
        :return:
        """

		net = self.build_net2()#self.build_net_noRelU()#
		print("### net created successfully ###")

		# advanced cost
		cost_average_pixels = tf.reduce_mean(tf.square(self.label - self.out))

		cost_similar_filters = tf.reduce_mean(tf.square(tf.stack(self.gradients)
								- tf.reduce_mean(tf.stack(self.gradients))))

		cost = cost_average_pixels #+ self.alpha*cost_similar_filters


		optimizer = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(cost)
		saver = tf.train.Saver()


		loss_graph = []
		loss_a = []
		loss_b = []

		with tf.Session() as sess:
			print("### started initialize variables ###")
			sess.run(tf.global_variables_initializer())
			print("### finished initialize variables ###")


			gen = utils.get_example(self.path, self.num_images)

			# n = [n.name for n in tf.get_default_graph().as_graph_def().node]
            #
			# gname = "graph 0"
			# f = open(gname, 'w')
			# for st in n:
			# 	f.write(st)
			# 	f.write("\n")
			# f.close()

			for iter_index in range(self.num_epochs):


				for index in range(self.num_images):

					(x, y) = next(gen)
					fd = {}

					for i in range(len(x[0])):
						fd[self.convs[i]] = x[0][i]
					for i in range(len(x[1])):
						fd[self.pool_dims[i]] = x[1][i]#[fd[self.convs[
							# i+1]].shape[1],
												# fd[self.convs[i + 1]].shape[2]]
						# print("fd[self.pool_dims[i]]: ", fd[self.pool_dims[i]])
						#fd[self.pool_ind[i]] = x[2][i]#sess.run(x[2][i])
						#print("fd[self.pool_ind[i]].shape: ", i, fd[self.pool_ind[
							# i]].shape)
					fd[self.label] = y


					_, c, c_a, c_b, ggg = sess.run([optimizer, cost,
										  cost_average_pixels,
									 cost_similar_filters, self.gradients],
									feed_dict=fd)
					# _, c = sess.run([optimizer, cost],
					# 				feed_dict=fd)


					# {self.convs: x[0],
					#  self.pool_dims: x[1],
					#  self.pool_ind: x[2],
					#  self.label: y}

					loss_graph.append(c)

				if iter_index % 50 == 0:
				# 	loss_graph.append(c)
					print("Finished iteration " + str(
						iter_index + 1) + " out of " +
						  str(self.num_epochs) + " with loss: " + str(c))
					print("pixelwise: %d", c_a)
					loss_a.append(c_a)
					print("gradients: %d", c_b)
					loss_b.append(c_b)
					print(ggg)

				# 	p = sess.run(self.out, feed_dict=fd)
				# 	im = p[0,:,:,:]
				# 	np.save("npiter_" +str(iter_index+1) + ".npy", im)
				# 	#scipy.misc.imsave("iter_" + str(iter_index+1) + ".jpg",
				# 	#  im)
				# 	matplotlib.image.imsave("iter_" + str(iter_index+1) + ".png", im)


			# n = [n.name for n in tf.get_default_graph().as_graph_def().node]
                #
				# gname = "graph" + str(iter_index+1)
				# f = open(gname, 'w')
				# for st in n:
				# 	f.write(st)
				# 	f.write("\n")
				# f.close()
                #
                #
				# print("names: ", len(n))

			writer = tf.summary.FileWriter("/cs/labs/raananf/rhomburger/code/Restoration/tmp/test", sess.graph)



			print("### finished training ###")

			np.save("loss_a.npy", np.array(loss_a))
			np.save("loss_b.npy", np.array(loss_b))

			save_path = saver.save(sess, self.model_location)
			print("Model saved in file: %s" % save_path)

			print("### finished saving the net ###")
			# tt = np.arange(len(loss_graph))
			# plt.scatter(tt, loss_graph)
			# #plt.savefig("loss_" + self.model_location + ".jpg")
			# plt.show()

			# plt.plot(np.arange(0,self.num_epochs,
			# 				   self.num_epochs/len(loss_graph)), loss_graph)



			#plottin error graph
			# plt.plot(np.arange(0,len(loss_graph)), loss_graph)
			# plt.show()






	def restore(self, features):
		# Gets array of responses from VGG (as used in Gatys' algorithm)
		# Returns an image J so that VGG(J) has the nearest responses to the
		# input responses
		net = self.build_net2()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, self.model_location)
			print("### model restored ###")

			fd = {}
			for i in range(len(features[0])):
				fd[self.convs[i]] = features[0][i]  # sess.run(x[0][i])
			# print("fd[self.convs[i]]: ", i, fd[self.convs[i]].shape)
			for i in range(len(features[1])):
				fd[self.pool_dims[i]] = features[1][i]  # [fd[self.convs[
				# i+1]].shape[1],
				# fd[self.convs[i + 1]].shape[2]]
				# print("fd[self.pool_dims[i]]: ", fd[self.pool_dims[i]])
				#fd[self.pool_ind[i]] = features[2][i]  # sess.run(x[2][i])


			blue, green, red = tf.split(axis=3, num_or_size_splits=3,
										  value=self.out)
			rgb = tf.concat(axis=3, values=[
				red + VGG_MEAN[2],
				green + VGG_MEAN[1],
				blue + VGG_MEAN[0],

			])

			#prediction = sess.run(rgb, feed_dict=fd)
			prediction = sess.run(self.out, feed_dict=fd)

		return prediction


#TODO a function that gives the weights
