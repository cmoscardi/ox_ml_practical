import sys, numpy, inspect


def todo():
  print "You should implement this function: %s" % inspect.stack()[1][3]
  sys.exit(1)


"""
Split data into 'splits' different sub-blocks, returning the
block indexed by 'test_split' plus the rest of the data.
"""
def split_data(splits, test_split, data):
  assert test_split >= 0 and test_split < splits
  assert splits <= len(data)

  split_size = len(data) / int(splits)
  test_start = split_size*test_split
  test_end   = test_start + split_size

  test  = data[test_start:test_end]
  train = numpy.concatenate([data[0:test_start],data[test_end:]])

  return (train,test)


"""
Python re-implementation of Matlab tiedrank function 
"""
def tiedrank(X):  
  Z = [(x, i) for i, x in enumerate(X)]  
  Z.sort()  
  n = len(Z)  
  Rx = [0]*n   
  start = 0 # starting mark  
  for i in range(1, n):  
     if Z[i][0] != Z[i-1][0]:
       for j in range(start, i):  
         Rx[Z[j][1]] = float(start+1+i)/2.0;
       start = i
  for j in range(start, n):  
    Rx[Z[j][1]] = float(start+1+n)/2.0;

  return Rx


"""
Python re-implementation of Kaggle's Matlab AUC code
"""
def AUC(labels, posterior):
  r = tiedrank(posterior)
  auc = (sum(r*(labels==1)) - sum(labels==1)*(sum(labels==1)+1)/2) / (sum(labels<1)*sum(labels==1));
  return auc
