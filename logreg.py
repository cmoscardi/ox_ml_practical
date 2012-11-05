import numpy as np
from util import *

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def log_sigmoid(z):
  # n.b. -np.logaddexp(0,-z) calculates -log(1+exp(-z)) 
  # in log space without exponentiation to avoid overflow.
  # Try help(numpy.logaddexp) in the interpreter for
  # more information.
  return -np.logaddexp(0,-z)

def log_sigmoid_complement(z):
  return -np.logaddexp(0,z)

# This function should calculate the negative conditional 
# log probability of the data x given weights w and 
# observed response variables y.
def objective(x, y, w):
  z = np.dot(x, w)
  dudes = map((lambda a: if_else(a[1]==1,log_sigmoid(a[0]),log_sigmoid_complement(a[0]))),zip(z,y))

  return reduce((lambda a,b: a+b), dudes)

def if_else(condition,a,b):
  if condition:
    return a
  else:
    return b

# This is the log Gaussian prior 
def log_prior(w, alpha):
  return np.dot(w, w) / (2*alpha)

# This function should calculate the gradient of the negative 
# conditional log probability of the data x given weights w
# and observed response variables y.
def grad(x, y, w):
  to_be_summed = map((lambda a: [sigmoid(np.dot(w,a[0])) - a[1] * l for l in a[0]]),zip(x,y))
  return reduce((lambda a,b: map(sum, zip(a,b))),to_be_summed)


# This is the derivative of the Gaussian prior
def prior_grad(w, alpha):
  return [ (i/alpha) for i in w]


