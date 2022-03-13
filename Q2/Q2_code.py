import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import math

input_colors = np.array([
[255,0,0],
[0,255,0],
[0,0,255],
[255,255,0],
[255,0,255],
[0,255,255],
[128,128,0],
[128,0,128],
[0,128,128],
[255,128,0],
[255,0,128],
[128,255,0],
[0,255,128],
[128,0,255],
[0,128,255],
[255,20,147],
[220,20,60],
[255,51,51],
[255,153,51],
[255,255,51],
[51,255,51],
[153,255,51],
[51,255,153],
[51,255,255]])

# Normalize the input (min-max normalize, but we already know the minimum and maximum)
norm_colors = input_colors/255

def sigma(k):
    return 18*math.exp(-1*k/1000.0)

def alpha(k):
    return 0.8*math.exp(-1*k/1000.0)

def neighbourhood(k, d_squared):
    return math.exp((-1.0*d_squared)/(2.0*pow(sigma(k), 2)))

# Initialize the system
space_size = 100 # 100 x 100 grid of neurons
max_epochs = 1000

# Initialize random weights with dim of 3 for RGB
w = np.random.random((space_size,space_size,3))
# diff = np.abs(np.sum(normRGB[0] - w, axis=2))
plt.imshow(w)
plt.show()

epoch = 0
while epoch <= max_epochs:
    
    for x in norm_colors:
        # calculate performance index
        diff = np.linalg.norm(x - w, axis =2)
        # find index of winning node
        ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape)

        # get neigbhourhood
        neighbourhood_val = 1
        Nc = 0
        while neighbourhood_val >= 0.5:
            d_squared = pow(Nc, 2)
            neighbourhood_val = neighbourhood(epoch, d_squared)
            if neighbourhood_val >= 0.5:
                Nc += 1

        # Update weights for neighbourhood
        for i in range(ind[0]-Nc, ind[0]+Nc+1):
            for j in range(ind[1]-Nc, ind[1]+Nc+1):
                if i >= 0 and j>=0 and i < space_size and j < space_size:
                    # make sure you don't exceed the size of the space
                    w[i][j] += alpha(epoch) * (x-w[i][j])

    plot_ind = [20, 40, 100, 1000]
    if epoch in plot_ind:
        print("Epoch Number: {}".format(epoch))
        plt.imshow(w)
        plt.show()
        
    epoch += 1