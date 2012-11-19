import numpy as np

#these are the lambdas, not the weight vector
lambdas = np.zeros(4)

#1st column is x0
#2nd column is x1
#3rd column is t
data = np.array([[1,1,-1],[1,0,1],[0,1,1],[0,0,-1]])

#splitting data into 
#features and output
x = data[:,:2]
t = data[:,2]



#kernel flag
LINEAR=True




def sub_iteration():
    #have to iterate because i need the
    #result after each single iteration
    
    for i in range(4):
        print "sub-iteration %s " % (i+1)
        y = sum(lambdas * t * kernel(x[i]))

        print "y was %s" %y
        print "t was %s" %t[i]
        if t[i]*y > 0:
            print "sub-iteration %s required no update" % (i+1)

        else:
            print "sub-iteration %s failed" % (i+1)
            print 'adding 1 to that weight'
            lambdas[i]+=1
            print 'new weights: \n%s' % lambdas

'''this calculates the kernel 
function with respect to one particular
training instance x_j, and returns a vector
of kernel(x_i, x_j) for all i.'''

def kernel(v):
    if LINEAR:
        kernel = np.dot(x,v)
        print "kernel is %s" % kernel
        return kernel
    else:
        kernel = 1 + np.dot(x,v)
        kernel = kernel*kernel
        print "poly kernel is %s" % kernel
        return kernel


#run script
for j in range(20):
    print '======BEGIN ITERATION %s ========' % (j+1)
    sub_iteration()
    print '======END ITERATION %s =======' % (j+1)

