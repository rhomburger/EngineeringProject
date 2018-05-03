import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import scipy.misc
import matplotlib.image




VGG_MEAN = [103.939, 116.779, 123.68]



class Restoration:

	def __init__(self, im_path, model_path, num_epochs, batch_size,f_size=3,
				 reg_factor=1, learning_rate = 0.001):

		self.path = im_path
		self.model_location = model_path
		self.filter_size = f_size # size of deconvolution filter

		self.weights = [] # dictionary of weights for each deconvolution layer
		self.biases = [] # dictionary of biases for each deconvolution layer

		self.convs = []
		self.pool_dims = [] # placeholders for original pre-pooling dims
		self.pool_ind = [] # placeholders for pooling argmax

		self.layers = []
		
		self.alpha = reg_factor #regularization factor of the loss function
		self.num_epochs = num_epochs
		self.num_images = batch_size

		self.random_weights = 0.05
		self.random_biases = 0
		self.learning_rate = learning_rate

		self.gradients = []

	def deconv(self, in_block, weights, name):
		"""
		wraper to deconvolution layer
		"""
		return tf.nn.conv2d(input=in_block, filter=weights,
								strides=[1,1,1,1], padding="SAME", name=name)

	def crop_and_concat(self, x1,x2):
		x1_shape = tf.shape(x1)
		x2_shape = tf.shape(x2)
		# offsets for the top left corner of the crop
		offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
		size = [-1, x2_shape[1], x2_shape[2], -1]
		x1_crop = tf.slice(x1, offsets, size)
		return tf.concat([x1_crop, x2], 3)

	def unpooling_zero_neighbours(self, updates, dim):
		"""
		Unpooling by doubling the shape. pixel (i, j) is now in (2i,2j) and
		its new neighbours are zeros
		"""
		updates_shape = tf.shape(updates)
		shape = tf.stack([1, dim[0], dim[1], updates_shape[3]])
		N = updates_shape[1]
		M = updates_shape[2]

		axis1 = tf.tile(2*tf.range(N), [M])
		axis1 = tf.reshape(axis1, [M,N])
		axis1 = tf.transpose(axis1)
		axis1 = tf.reshape(axis1, [N*M])

		axis2 = tf.tile(2*tf.range(M),[N])

		indices = tf.concat([[tf.zeros([M*N], dtype=tf.int32)], [axis1], \
															 [axis2]], axis=0)
		indices = tf.transpose(indices)
		indices = tf.reshape(indices, [N, M,3])
		indices = tf.expand_dims(indices,0)

		scatter = tf.scatter_nd(indices, updates, shape)
		return scatter

	def unpooling(self, inp, layer=-1):
		"""
		wraper for unpooling layer
		"""
		dims = tf.placeholder("int32", shape=[2])
		self.pool_dims.append(dims)
		return self.unpooling_zero_neighbours(inp, dims)

	def build_net(self):
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

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,512,512]),
									  name="weights1"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			512]), name="biases1"))
		t = tf.nn.relu(self.deconv(self.convs[0], self.weights[0], "deconv1") +
									   self.biases[0], name="relu1")
		t = self.unpooling(t, 4)

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*512,256]),
									  name="weights2"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			256]), name="biases2"))

		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[1]], 3),
								   self.weights[1], "deconv2") +
					   self.biases[1], name="relu2")
		t = self.unpooling(t, 3)

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*256,128]),
									  name="weights3"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			128]), name="biases3"))
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[2]], 3),
								   self.weights[2], "deconv3") +
					   self.biases[2], name="relu3")
		t = self.unpooling(t, 2)

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*128,64]),
										name="weights4"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			64]), name="biases4"))
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[3]], 3),
								   self.weights[3], "deconv4") +
					   self.biases[3], name="relu4")
		t = self.unpooling(t, 1)

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
							self.filter_size,self.filter_size,2*64,64]),
									  name="weights5"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			64]), name="biases5"))
		t = tf.nn.relu(self.deconv(tf.concat([t, self.convs[4]], 3),
								   self.weights[4], "deconv5") +
					   self.biases[4], name="relu5")

		self.weights.append(tf.Variable(self.random_weights*tf.random_normal([
			self.filter_size, self.filter_size, 64, 3]), name="weights6"))
		self.biases.append(tf.Variable(self.random_biases*tf.random_normal([
			3]), name="biases6"))
		self.out = self.deconv(t, self.weights[5], "deconv6-out") + \
				   self.biases[5]

		for i in range(5):
			self.gradients.append(tf.sqrt(tf.reduce_sum(tf.square(
				tf.gradients(self.out, self.convs[i])))))

		return self.out

	def train_net(self):
		"""
        trains the restoration net
        """
		net = self.build_net()
		print("### net created successfully ###")

		# cost function
		cost_average_pixels = tf.reduce_mean(tf.square(self.label - self.out))

		cost_similar_filters = tf.reduce_mean(tf.square(tf.stack(
			self.gradients) - tf.reduce_mean(tf.stack(self.gradients))))
		cost = cost_average_pixels + self.alpha*cost_similar_filters

		optimizer = tf.train.AdamOptimizer(
			learning_rate=self.learning_rate).minimize(cost)
		saver = tf.train.Saver()

		loss_graph = []
		loss_a = []
		loss_b = []
		loss_g = []

		with tf.Session() as sess:
			print("### initializing variables ###")
			sess.run(tf.global_variables_initializer())

			gen = utils.get_example(self.path, self.num_images)

			print("### strating the training ###")

			for iter_index in range(self.num_epochs):
				for index in range(self.num_images):
					(x, y) = next(gen)
					fd = {}

					for i in range(len(x[0])):
						fd[self.convs[i]] = x[0][i]
					for i in range(len(x[1])):
						fd[self.pool_dims[i]] = x[1][i]
					fd[self.label] = y

					_, c, c_a, c_b, c_g = sess.run([optimizer, cost,
										  cost_average_pixels,
									 cost_similar_filters, self.gradients],
									feed_dict=fd)

					loss_graph.append(c)
					loss_a.append(c_a)
					loss_b.append(c_b)
					loss_g.append(c_g)

				if iter_index % 50 == 0 or iter_index == self.num_epochs-1:
					print("Finished iteration " + str(
						iter_index + 1) + " out of " +
						  str(self.num_epochs) + " with loss: " + str(c))
					print("pixelwise: %d", c_a)
					print("gradients: %d", c_b)
					print(c_g)

			print("### finished training ###")


			save_path = saver.save(sess, self.model_location)
			print("Model saved in file: %s" % save_path)

			np.save("stats/loss_total.npy", loss_graph)
			np.save("stats/loss_a.npy", loss_a)
			np.save("stats/loss_b.npy", loss_b)
			np.save("stats/loss_g.npy", loss_g)

	def restore(self, features, debug=False):
		# Gets array of features (responses from VGG + dimensions)
		# Returns an image J so that VGG(J) has the nearest responses to the
		# input responses
		net = self.build_net()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess, self.model_location)

			fd = {}
			for i in range(len(features[0])):
				fd[self.convs[i]] = features[0][i]
			for i in range(len(features[1])):
				fd[self.pool_dims[i]] = features[1][i]

			prediction, gradients = sess.run([self.out, self.gradients],
									   feed_dict=fd)
			if debug:
				print(gradients)

		return prediction

