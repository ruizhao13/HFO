import tensorflow as tf 
import numpy as np
import math
from functools import partial

# Hyper Parameters
LAYER1_SIZE = 1024
LAYER2_SIZE = 512
LAYER3_SIZE = 256
LAYER4_SIZE = 128
LEARNING_RATE = 1e-3
TAU = 0.001
BATCH_SIZE = 64

class ActorNetwork:
	"""docstring for ActorNetwork"""
	def __init__(self,sess,state_dim,action_dim):

		self.sess = sess
		self.state_dim = state_dim
		self.action_dim = action_dim
		
		
		self.action_up_bound = np.array([100., 100., 100., 100., 100., 180., 180., 180., 100., 180.], dtype=np.float)
		self.action_bottom_bound = np.array([-100., -100., -100., -100., -100., -180., -180., -180., 0., -180.], dtype=np.float)
		self.action_bound = tf.constant([100., 100., 100., 100., 100., 180., 180., 180., 50., 180.], dtype=tf.float32)
		self.action_bias = tf.constant([0., 0., 0., 0., 0., 0., 0., 0., 50., 0.], dtype=tf.float32)


		# create actor network
		self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim)

		# create target actor network
		self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net)
		
		# define training rules
		self.create_training_method()

		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		#self.load_network()

		
	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
		self.parameters_gradients = tf.gradients(self.action_output,self.net,self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

	def create_network(self,state_dim,action_dim):
		layer1_size = LAYER1_SIZE
		layer2_size = LAYER2_SIZE
		layer3_size = LAYER3_SIZE
		layer4_size = LAYER4_SIZE

		state_input = tf.placeholder("float",[None,state_dim])

		W1 = self.variable([state_dim,layer1_size],state_dim)
		b1 = self.variable([layer1_size],state_dim)
		W2 = self.variable([layer1_size,layer2_size],layer1_size)
		b2 = self.variable([layer2_size],layer1_size)
		W3 = self.variable([layer2_size, layer3_size], layer2_size)
		b3 = self.variable([layer3_size], layer3_size)
		W4 = self.variable([layer3_size, layer4_size], layer3_size)
		b4 = self.variable([layer4_size], layer4_size)

		W5 = tf.Variable(tf.random_uniform([layer4_size,action_dim],-3e-3,3e-3))
		b5 = tf.Variable(tf.random_uniform([action_dim],-3e-3,3e-3))

		
		
		
		
		
		
		
		
		layer1 = tf.nn.leaky_relu(tf.matmul(state_input,W1) + b1, alpha=0.01)
		layer2 = tf.nn.leaky_relu(tf.matmul(layer1,W2) + b2, alpha=0.01)
		layer3 = tf.nn.leaky_relu(tf.matmul(layer2,W3) + b3, alpha=0.01)
		layer4 = tf.nn.leaky_relu(tf.matmul(layer3,W4) + b4, alpha=0.01)
		

		# action_output_norm = tf.tanh(tf.matmul(layer4,W5) + b5)
		# action_output = tf.add(tf.multiply(action_output_norm, self.action_bound), self.action_bias)
		action_output = tf.matmul(layer4, W5) + b5
		return state_input,action_output,[W1,b1,W2,b2,W3,b3,W4,b4,W5,b5]

	def create_target_network(self,state_dim,action_dim,net):
		state_input = tf.placeholder("float",[None,state_dim])
		ema = tf.train.ExponentialMovingAverage(decay=1-TAU)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(state_input,target_net[0]) + target_net[1])
		layer2 = tf.nn.relu(tf.matmul(layer1,target_net[2]) + target_net[3])
		
		
		
		layer1 = tf.nn.leaky_relu(tf.matmul(state_input,target_net[0]) + target_net[1], alpha=0.01)
		layer2 = tf.nn.leaky_relu(tf.matmul(layer1,target_net[2]) + target_net[3], alpha=0.01)
		layer3 = tf.nn.leaky_relu(tf.matmul(layer2,target_net[4] + target_net[5]), alpha=0.01)
		layer4 = tf.nn.leaky_relu(tf.matmul(layer3,target_net[6] + target_net[7]), alpha=0.01)
		# action_output_norm = tf.tanh(tf.matmul(layer4,target_net[8]) + target_net[9])
		# action_output = tf.add(tf.multiply(action_output_norm, self.action_bound), self.action_bias)
		action_output = tf.matmul(layer4,target_net[8]) + target_net[9]

		return state_input,action_output,target_update,target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self,q_gradient_batch,state_batch):
		action_batch = self.actions(state_batch)
		with open('/home/ruizhao/Desktop/a.txt', 'a') as f:
			print("action_batch[0]2", file=f)
			print(action_batch[0], file=f)
		# print("action")
		# print(action_batch[0])
		# print("gradient")
		# print(q_gradient_batch)
		# print(q_gradient_batch)
		q_gradient_batch_abs = np.absolute(q_gradient_batch)
		a = q_gradient_batch_abs.max(axis=1)
		# print('q_max')
		# print(a)
		b = np.resize(a, [len(action_batch), 1])
		# q_gradient_batch = q_gradient_batch / b
		with open('/home/ruizhao/Desktop/a.txt', 'a') as f:
			print("gradient_norm", file=f)
			print(q_gradient_batch[0], file=f)
		# print("gradient_xishu")
		# for i in range(self.action_dim):
			# print((self.action_up_bound[i] - action_batch[0][i]) / (self.action_up_bound[i] - self.action_bottom_bound[i]), end='  ')
		for _, q_gra in enumerate(q_gradient_batch):
			for i in range(4):
				if q_gra[i] > 0:
					if(q_gra[i] + action_batch[_][i] > self.action_up_bound[i]):
						q_gra[i] = 0
				else :
					if(q_gra[i] + action_batch[_][i] < self.action_bottom_bound[i]):
						q_gra[i] = 0
			for i in range(4, self.action_dim):
				if q_gra[i] > 0:
					q_gradient_batch[_][i] = q_gradient_batch[_][i] * (self.action_up_bound[i] - action_batch[_][i]) / (self.action_up_bound[i] - self.action_bottom_bound[i])
					with open('/home/ruizhao/Desktop/a.txt', 'a') as f:
						print()
					# if _==0:
						# print((self.action_up_bound[i] - action_batch[0][i]) / (self.action_up_bound[i] - self.action_bottom_bound[i]), end='  ')
				else:
					q_gradient_batch[_][i] = q_gradient_batch[_][i] * (action_batch[_][i] - self.action_bottom_bound[i]) / (self.action_up_bound[i] - self.action_bottom_bound[i])
					# if _==0:
						# print((action_batch[_][i] - self.action_bottom_bound[i]) / (self.action_up_bound[i] - self.action_bottom_bound[i]), end='  ')
		# print()
		with open('/home/ruizhao/Desktop/a.txt', 'a') as f:
			print("gradient_changed",file=f)
			print(q_gradient_batch[0], file=f)
		self.sess.run(self.optimizer,feed_dict={
			self.q_gradient_input:q_gradient_batch,
			self.state_input:state_batch
			})

	def actions(self,state_batch):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:state_batch
			})

	def action(self,state):
		return self.sess.run(self.action_output,feed_dict={
			self.state_input:[state]
			})[0]


	def target_actions(self,state_batch):
		return self.sess.run(self.target_action_output,feed_dict={
			self.target_state_input:state_batch
			})

	# f fan-in size
	def variable(self,shape,f):
		return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
	def load_network(self):
		self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
			print "Could not find old network weights"
	def save_network(self,time_step):
		print 'save actor-network...',time_step
		self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''

		
