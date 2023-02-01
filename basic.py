import numpy as np
import math
import matplotlib.pyplot as plt
from helpers import *
import pickle

# class Neural_Net:
#     def __init__(self, shape):
#         self.shape = shape
#         self.weights = []

A = [ # ONLY MATRICES NOT VECTORS
np.array([[5], [-5], [5] ,[-5]]), 
np.array([
[10, 10, 0, 0],
[0, 0, 10, 10]
]),
np.array([[10.902, 9.223]])
] #  Hidden layer connections

B = [
np.array([[-0.75], [1.751], [-3.25], [4.25]]),
np.array([[-12.5], [-12.5]]),
np.array([[-2]])
] #  Bias connections

A = load("trained1")[0]
B = load("trained1")[1]

connections = len(A)
layers = connections + 1

def makenet(weights, biases):
    def net(input: np.ndarray):
        s = [input]# Neuron Input - post normalization
        h = [input] # Neuron Output
        for i in range(connections):
            s.append(nextlayer(weights[i], h[i], biases[i]))
            h.append(sig(s[i+1]))
        return (s,h)
    return net

learning = 0.01
def backprop(data, iterations, progress=False, rsses=False):
    weights = copyarr(A)
    biases = copyarr(B)
    for i in range(iterations):
        rss = 0
        net = makenet(weights, biases)
        for pt in data:
            netstate = net(np.array([[pt[0]]]))
            sums = netstate[0]
            activations = netstate[1]

            # dW = [copyshape(A) for A in weights]
            # dB = [copyshape(B) for B in biases]
            x = pt[0]
            y = pt[1]
            out = activations[-1] # output
            rss += (out - np.array([[y]]))**2
            av_partials = [2 * (out - np.array([[y]]))] # check format of dataset so that it is scalar!!
            bs_partials = [] # trim last one
            wt_partials = [] # trim first one

            for j in range(layers)[::-1]:
                thislayer = elemwise(av_partials[0], sigprime(sums[j])) # also equal to dRSS/dB_l, so
                bs_partials.insert(0, thislayer)
                weightsbefore = np.outer(thislayer, activations[max(j-1,0)])
                wt_partials.insert(0, weightsbefore)

                transpose = weights[max(j-1, 0)].transpose()
                prevlayer = matmul(transpose, thislayer)
                av_partials.insert(0, prevlayer)
            bs_partials.pop(0) 
            wt_partials.pop(0) 

            # for i in range(len(weights)):
            #     print(weights[i].shape, wt_partials[i].shape)
            weights = adjust_list(weights, wt_partials, learning)
            # print(weights, 2)
            # print(biases, bs_partials)
            biases = adjust_list(biases, bs_partials, learning)

            # Progress
        if rsses:
            print("RSS:", rss.item())
        if progress and (i / iterations)*100 % 1 == 0:
            print(i * 100 / iterations, "%")

    print("100.0 % - Finished")
    return (weights, biases)
        
        


dataset = [(0,0),(0.25,1),(0.5,0.5),(0.75,1),(1,0)]


trained = backprop(dataset, 100000, True)

Ao = trained[0]
Bo = trained[1]
save(trained, "trained1")

initialnet = makenet(A, B)
plot(dataset, initialnet, 1000, "r")
trainednet = makenet(Ao, Bo)
plot(dataset, trainednet, 1000)


plt.show()