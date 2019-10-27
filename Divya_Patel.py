## ---------------------------- ##
##
## Example student submission code for autonomous driving challenge.
## You must modify the train and predict methods and the NeuralNetwork class.
##
## ---------------------------- ##



#Autonomus Car - Divya Patel

import numpy as np
import cv2
from tqdm import tqdm
import time
from math import sqrt
from scipy import ndimage


def train(path_to_images, csv_file):
    data = np.genfromtxt(csv_file, delimiter = ',')
    frame_nums = data[:,0]
    steering_angles = data[:,1]

    training_angles = convert_to_array(steering_angles)

    training_angles = [gaussian_filter(i) for i in training_angles]
    # training_angles = []
    # for i in steering_angles:
    #     training_angles.append([i])
    # training_angles = np.array(training_angles)
    # You could import your images one at a time or all at once first,
    # here's some code to import a single image:

    training_im = []

    for i in range(len(frame_nums)):
        frame_num = int(frame_nums[i])
        im_full = cv2.imread(path_to_images + '/' + str(int(frame_num)).zfill(4) + '.jpg')
        im_resize = cv2.resize(im_full, (60,64))
        #print(np.array(im_resize[20:]>100).astype('uint8'))
        #im_rb = im_rb/255.
        # im_resize = im_resize/255
        # im_rb = im_resize
        #im_rb = rbi(im_resize)
        #print(im_rb)
        im_resize = cv2.cvtColor(im_resize, cv2.COLOR_BGR2GRAY)
        im_rb = np.array(im_resize[30:]>250).astype('uint8')

        training_im.append(im_rb.ravel())
    training_im = np.array(training_im)


    NN = NeuralNetwork()
    params = NN.getParams()


    yHat = NN.forward(training_im)
    grads = NN.computeGradients(training_im,training_angles)

    ar = 0.01
    b,b1,epsilon =0.9,0.999,1e-08
    iters = 2000

    m,v = np.ones(len(grads)),np.ones(len(grads))

    J_list = []
    temp = 0

    for _ in tqdm(range(iters)):
        temp += 1
        g = NN.computeGradients(training_im, training_angles)
        m = b * m + (1 - b) * g
        v = b1 * v + (1 - b1) * np.power(g, 2)
        m_hat = m / (1 - np.power(b, temp))
        v_hat = v / (1 - np.power(b1, temp))
        NN.setParams(NN.getParams() - ar * m_hat / (np.sqrt(v_hat) + epsilon))
        J_val = NN.costFunction(training_im,training_angles)
        #print(J_val)

        J_list.append(J_val)
    #print(J_list)
    #print(NN.getParams())
    # yHat = NN.forward(training_im)
    # print(yHat.shape, training_angles.shape)
    # print("RMSE ", np.sqrt(np.mean(np.array(b[1][np.argmax(yHat)] - b[1][np.argmax(y)])**2)))
    #T = trainer(NN)
    #T.train(training_im,training_angles)


    return NN

#print("bbbbbbbbbbbbb", b)
def predict(NN, image_file):
    '''
    Second method you need to complete.
    Given an image filename, load image, make and return predicted steering angle in degrees.
    '''
    im_full = cv2.imread(image_file)
    im_resize = cv2.resize(im_full, (60,64))
    #print(np.array(im_resize[20:]>100).astype('uint8'))
    #im_rb = im_rb/255.
    # im_resize = im_resize/255
    # im_rb = im_resize
    #im_rb = rbi(im_resize)
    #print(im_rb)
    im_resize = cv2.cvtColor(im_resize, cv2.COLOR_BGR2GRAY)
    im_rb = np.array(im_resize[30:]>250).astype('uint8')
    y = NN.forward(im_rb.ravel())

    #return y[0]
    #angle = convert_to_angle(y)

    return convert_to_angle_1(y)
    #print(b[1][max_idx],convert_to_angle(yHat))
    #print("---------------------------------------------------")


def convert_to_array(steering_angles):
    angles_training = []
    b = np.histogram(0,range=(-180,120), bins=64)
    for i in range(len(steering_angles)):
        t = np.histogram(steering_angles[i],b[1])[0].astype('float')
        angles_training.append(t)
        #print(training_angles[0],training_angles[-1],steering_angles[0],steering_angles[-1])
    return np.array(angles_training)


def gaussian_filter(arr):
    return ndimage.gaussian_filter1d(np.float_(arr), 1)

def convert_to_angle(arr):
    centers = np.linspace(-180,120,64)
    return centers[np.argmax(arr)]

def convert_to_angle_1(y):
    b = np.histogram(0,range=(-180,120), bins=64)
    return b[1][np.argmax(y)]


def rbi(im):
    return im[:,:,0]*1+ im[:,:,1]*0 + im[:,:,2]*-1



class NeuralNetwork(object):
    def __init__(self):
        '''
        Neural Network Class, you may need to make some modifications here!
        '''
        self.inputLayerSize = 60*34
        self.outputLayerSize = 64
        self.hiddenLayerSize = 30

        #Weights (parameters)
        # self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        # self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

        limit = sqrt(6 / (self.inputLayerSize + self.hiddenLayerSize))
        self.W1 = np.random.uniform(-limit, limit, (self.inputLayerSize, self.hiddenLayerSize))

        limit = sqrt(6 / (self.hiddenLayerSize + self.outputLayerSize))
        self.W2 = np.random.uniform(-limit, limit, (self.hiddenLayerSize, self.outputLayerSize))

    def forward(self, X):
        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        # yHat = self.z3
        return yHat

    def sigmoid(self, z):
        #return np.exp(z)/np.sum(np.exp(z))
        #Apply sigmoid activation function to scalar, vector, or matrix
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        #return np.exp(z)/np.sum(np.exp(z))
        #return np.ones(z.shape)
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        #print(self.yHat.shape)
        #print(y.shape)
        J = 0.5*np.sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        self.yHat = self.forward(X)
        #print(self.yHat.shape,y.shape,self.sigmoidPrime(self.z3).shape)
        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2


    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)

        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': 5000, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS',args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
