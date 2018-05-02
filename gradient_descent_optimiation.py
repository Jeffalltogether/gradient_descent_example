# https://spin.atomicobject.com/2014/06/24/gradient-descent-linear-regression/
# convex optimiation of a MSE function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sympy import Symbol, Derivative

def mean_sq_error(y, fun, *args):
	try:
		return 1.0/len([y]) * np.sum((y - fun(args)))**2

	except TypeError:
		return 1.0 * np.sum((y - fun(args)))**2

def linear_model(*args):
	x = args[0][0]
	m = args[0][1]
	b = args[0][2]

	return m*x + b

def gradients_linear_model(x, y, m, b, *args):
	try:
		m_gradient = 2.0/len(x) * np.sum(-x* (y - (m*x + b)))
		b_gradient = 2.0/len(x) * np.sum(-1.0 * (y - (m*x + b)))

	except TypeError:
		m_gradient = 2.0 * (-x * (y - (m*x + b)))
		b_gradient = 2.0 * (-1.0 * (y - (m*x + b)))

	return m_gradient, b_gradient

def plot_3d(X,Y,Z, *args):
	X, Y = np.meshgrid(X, Y)
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	# Plot the surface.
	# These must be true for 3-D plot to work
	# Z is a list of lists (2D-array)
	# X and Y are lists (1D-array)
	# len(X) == len(Y) == len(Z) == len(Z[n]), where n = len(Z)
	ax.plot_wireframe(X, Y, Z, linewidth=1, antialiased=True)

	# add gradient descent path
	if args:
		ax.plot(args[0], args[1], args[2], 'o-', color = 'black', linewidth = 4, markersize = 10)	
	# Customize the z axis.
	# ax.set_zlim(-10.01, 10.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# set x and y limits
	# ax.set_xlim(xmin = -200.0)
	# ax.set_ylim(ymin = -200.0)

	# Add a color bar which maps values to colors.
	# fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel('Slope', fontsize=18)
	ax.set_ylabel('Intercept', fontsize=18)
	ax.set_zlabel('Mean Square Error', fontsize=18)

	plt.show()
	return

def plot_scatter(x,y,*args):
	fig, ax = plt.subplots()
	ax.scatter(x,y)
	
	if args:
		try:
			len(args[1])
		except TypeError:
			y_pred = linear_model(args)
			ax.plot(args[0], y_pred, '-')
		else:
			alp = np.arange(0.0, 1.0, 1.0/len(args[1]))
			for i in range(len(args[1])):
				y_pred = linear_model((args[0], args[1][i], args[2][i]))
				ax.plot(args[0], y_pred, '-', color = 'blue')#, alpha = alp[i])
	
	plt.show()

	return

if __name__ == '__main__':

	# 1- generate data
	points_x = np.arange(1,10,0.4)
	points_x = points_x*np.random.normal(0.5,0.1,len(points_x))
	points_y = np.arange(2, 17, 0.68)
	points_y = points_y*np.random.normal(0.5,0.1,len(points_y))

	# plot_scatter(points_x,points_y)

	# 2.0 plot the decision surfaace
	# select a range of parameters for the slope and intercept of the model
	m = np.arange(-1000.0, 1000.0, 25.0)
	b = np.arange(-1000.0, 1000.0, 25.0)

	# compute the MSE at each slope-intercept combination
	error_matrix = []
	for slope in m:
		error_at_slope = []
		for intercept in b:
			error_at_slope.append( mean_sq_error(points_y, linear_model, points_x, slope, intercept) )
		error_matrix.append(error_at_slope)

	# plot the error surface at each slope-intercept combination
	# this is a positive semi-definite surface (x**2)
	# plot_3d(m,b,error_matrix)

	#2 - compute the partial derivatives of a function with respect to x and y.
	# see function called `gradients_linear_model`

	################### Batch Gradient Descent
	MSE = 1000.0
	# startParams 
	slope, intercept = [900.0, 900.0]
	batch_descent = []
	# while MSE >= 0.1:
	while MSE >= 15.0:
		print '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'

		#3 - compute the error of a function at x1, y1 at a reasonable set of starting parameters
		MSE = mean_sq_error(points_y, linear_model, points_x, slope, intercept)
		print 'Current MSE at slope %.2f and intercept %0.2f = %0.4f' %(slope, intercept, MSE)

		# if MSE <= 1000.0:
		batch_descent.append([slope, intercept, MSE])
		
		#4 - compute the gradients of the slope and intercept at the current slope and intercept values
		grad_m, grad_b = gradients_linear_model(points_x, points_y, slope, intercept)
		print 'The gradient of the slope is %.2f; and the gradient of the intercept is %0.2f' %(grad_m, grad_b)

		#5 - update the estimated slope and intercept values
		learning_rate = 0.08
		slope = slope - learning_rate * grad_m
		intercept = intercept - learning_rate * grad_b
		print 'the new slope is %.2f; and the new intercept is %0.2f' %(slope, intercept)

	batch_descent = np.array(batch_descent)
	batch_descent = np.reshape(batch_descent.T, (3, len(batch_descent)))

	plot_3d(m,b,error_matrix, batch_descent[1], batch_descent[0], batch_descent[2])

	x_plot = np.arange(points_x.min(), points_x.max(), 0.5)

	plot_scatter(points_x, points_y, x_plot, batch_descent[0][-1000:-1:100], batch_descent[1][-1000:-1:100])

	############# Stochastic Gradient descent
	# startParams 
	slope, intercept = [900.0, 900.0]
	stochastic_descent = []
	epochs = 100
	
	# loop through epochs
	for ep in range(epochs):
		# randomize order
		points = zip(points_x, points_y)
		points = np.random.permutation(points)

		# loop through points
		for obs in points:
			single_x,single_y = obs
			print '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
			#3 - compute the error of a function at x1, y1 at a reasonable set of starting parameters
			MSE = mean_sq_error(points_y, linear_model, points_x, slope, intercept)
			print 'Current MSE at slope %.2f and intercept %0.2f = %0.4f' %(slope, intercept, MSE)
			
			# if MSE <= 1000.0:
			stochastic_descent.append([slope, intercept, MSE])

			#4 - compute the gradients of the slope and intercept at the current slope and intercept values
			grad_m, grad_b = gradients_linear_model(single_x, single_y, slope, intercept)
			print 'The gradient of the slope is %.2f; and the gradient of the intercept is %0.2f' %(grad_m, grad_b)

			#5 - update the estimated slope and intercept values
			learning_rate = 0.08
			slope = slope - learning_rate * grad_m
			intercept = intercept - learning_rate * grad_b
			print 'the new slope is %.2f; and the new intercept is %0.2f' %(slope, intercept)

	stochastic_descent = np.array(stochastic_descent)
	stochastic_descent = np.reshape(stochastic_descent.T, (3, len(stochastic_descent)))

	plot_3d(m,b,error_matrix, stochastic_descent[1], stochastic_descent[0], stochastic_descent[2])

	x_plot = np.arange(points_x.min(), points_x.max(), 0.5)

	plot_scatter(points_x, points_y, x_plot, stochastic_descent[0][-1000:-1:100], stochastic_descent[1][-1000:-1:100])

	############# Mini-batch Gradient descent
	# startParams 
	slope, intercept = [900.0, 900.0]
	mini_batch_descent = []
	epochs = 100
	batch_size = 5
	
	# loop through epochs
	for ep in range(epochs):
		# randomize order
		points = zip(points_x, points_y)
		points = np.random.permutation(points)
		num_batches = len(points)//batch_size

		for batch_num in range(num_batches):
			# get point in batch and separate into x and y arrays
			batch = points[batch_num * batch_size :  (batch_num + 1) * batch_size]
			batch_x = np.array([p[0] for p in batch])
			batch_y = np.array([p[1] for p in batch])

			print '\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'

			#3 - compute the error of a function at x1, y1 at a reasonable set of starting parameters
			MSE = mean_sq_error(points_y, linear_model, points_x, slope, intercept)
			print 'Current MSE at slope %.2f and intercept %0.2f = %0.4f' %(slope, intercept, MSE)

			# if MSE <= 1000.0:
			mini_batch_descent.append([slope, intercept, MSE])
			
			#4 - compute the gradients of the slope and intercept at the current slope and intercept values
			grad_m, grad_b = gradients_linear_model(batch_x, batch_y, slope, intercept)
			print 'The gradient of the slope is %.2f; and the gradient of the intercept is %0.2f' %(grad_m, grad_b)

			#5 - update the estimated slope and intercept values
			learning_rate = 0.08
			slope = slope - learning_rate * grad_m
			intercept = intercept - learning_rate * grad_b
			print 'the new slope is %.2f; and the new intercept is %0.2f' %(slope, intercept)

	mini_batch_descent = np.array(mini_batch_descent)
	mini_batch_descent = np.reshape(mini_batch_descent.T, (3, len(mini_batch_descent)))

	plot_3d(m,b,error_matrix, mini_batch_descent[1], mini_batch_descent[0], mini_batch_descent[2])

	x_plot = np.arange(points_x.min(), points_x.max(), 0.5)

	plot_scatter(points_x, points_y, x_plot, mini_batch_descent[0][-1000:-1:100], mini_batch_descent[1][-1000:-1:100])