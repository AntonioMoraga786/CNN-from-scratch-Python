import random
import time
from math import log

class Conv():
    def __init__(self,kD,n,I):
        # function to intialize all the paramers of the layers
        # kD is a variablel with the dimensions of the kernel [x,y,z]
        self.n = n# number of neurons
        self.im = I# input image dimensions [x,y,z]
        self.outputD = [1+self.im[0]-self.k[0],1+self.im[1]-self.k[1],self.n]
        
        self.bias = [2*random.random()-1 for i in range(n)]

        ## initialize the weights
        self.k = kD# save the kernel dimensions
        self.kernel = []# initialize the kernel weights list

        # loop though every neuron
        for neuron in range(self.n):
            k = []# buffer for the whole kernel of a neuron
            # loop though all of the kernel values to initialize them
            for z in range(self.k[2]):
                plane = []# buffer for the weights of the kernel

                for y in range(self.k[1]):
                    row = []# buffer for the weights of the kernel

                    for x in range(self.k[0]):
                        # generate a random weight -1 to 1
                        w = 2*random.random()-1
                        row.append(w)

                    plane.append(row)
                k.append(plane)
            self.kernel.append(k)

    def Pass(self,image):
        # perform a forward pass
        self.input = image# store the input matrix as well

        # initialize output list
        self.output = [[[0 for i in range(self.outputD[0])] for i in range(self.outputD[1])] for i in range(self.outputD[2])]

        # loop though all the output values
        for z in range(self.n):# for every neuron
            for y in range(self.outputD[1]):
                for x in range(self.outputD[0]):
                    o = self.bias[z]

                    # loop though allthe values in convolution of this value
                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                o += self.input[kz][y+ky][x+kx]*self.kernel[z][kz][ky][kx]

                    self.output[z][y][x] = o

        # convolution is done

    def Back(self,dLdO):# perform back propagation
        ## initialize derivative lists
        self.dLdB = []#dL/dB

        ## calculate dL/dB:

        # for the dLdO of every kernel (2D)
        for kernel in dLdO:# for every neuron
            dLdB = 0# buffer for dL/dB for a kernel
            for row in kernel:# for the row in each plane
                dLdB += sum(row)# add the sum of all the values in the row

            self.dLdB.append(dLdB)# add value into bias derivatives

        ## v2
        self.dLdI = [[[0 for i in range(self.im[0])] for i in range(self.im[1])] for i in range(self.im[2])]
        self.dLdW = [[[[0 for i in range(self.k[0])] for i in range(self.k[1])] for i in range(self.k[2])] for i in range(self.n)]
        
        # loop though all the output values
        for z in range(self.outputD[2]):# for every neuron
            for y in range(self.outputD[1]):# for every row in each output
                for x in range(self.outputD[0]):

                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                self.dLdI[kz][y+ky][x+kx] += self.kernel[z][kz][ky][kx]*dLdO[z][y][x]
                                self.dLdW[z][kz][ky][kx] += self.input[kz][y+ky][x+kx]*dLdO[z][y][x]

class dense():
    def __init__(self,n,I):
        self.inputD = I# input dimension (a singular integer value)
        self.bias = [2*random.random()-1 for i in range(n)]
        self.outputD = n

        self.kernel = []
        # loop though every neuron and generate its kernel values
        for neuron in range(n):
            weights = [2*random.random()-1 for i in range(I)]

            self.kernel.append(weights)

    def Pass(self,Input):
        # initialize variables
        self.input = Input
        self.output = []

        # loop though every neuron and calculate the output
        for n in range(self.outputD):
            output = self.bias[n]# add the bias value first

            # loop though every pair of weight and input vals
            for w,i in zip(self.kernel[n],self.input):
                output += w*i

            self.output.append(output)# append the calculated value into the list of outputs


    def Back(self,dLdO):
        ## calculate dL/dB
        self.dLdB = dLdO# since each bias is used in one output and the derivative is equal to 1

        ## calculate dL/dI
        self.dLdI = []

        for i in range(self.inputD):
            dldi = 0# buffer for derivative of one input

            for n in range(self.outputD):# for every neuron and dLdO in the layer
                dldi += self.kernel[n][i]*dLdO[n]

            self.dLdI.append(dldi)

        ## calculate dLdW
        self.dLdW = []

        for n in range(self.outputD):# for every neuron in the layer
            dldw = []# buffer for the derivatives of the weights

            for i in range(self.inputD):# for every weight/input in each neuron
                dldw.append(self.input[i]*dLdO[n])

            self.dLdW.append(dldw)

class ReLU():
    def __init__(self,inputD):
        self.outputD = inputD# calculate the output dimensions

    def Pass(self,input):
        self.input = input# store the input values
        self.output = []

        # loop though every item in the input and calculate output
        if type(self.input[0]) == list:
            for item in self.input:
                self.output.append(self.recursion(item))

        else:
            self.output = self.recursion(self.input)

    def recursion(self,inv):
        # check if we are dealing with a miltidimensional list
        if type(inv[0]) != list:
            return [max(0,item) for item in inv]
        
        # we are dealing with a list, continue recursion
        else:
            out = [self.recursion(item) for item in inv]

            return out
        
    def Back(self,dLdO):
        self.dLdI = []# initialize dL/dI

        # loop though every item in the output and calculate derivative
        if type(self.output[0]) == list:
            for item,d in zip(self.output,dLdO):
                self.dLdI.append(self.Brecursion(item,d))

        else:
            self.dLdI = self.Brecursion(self.output,dLdO)

    def Brecursion(self,outv,der):
        if type(outv[0]) != list:# we are not dealing with a list anymore
            return [d if v >0 else 0 for v,d in zip(outv,der)]
        
        # we are dealing with a list, continue recursion
        else:
            out = [self.Brecursion(item,d) for item,d in zip(outv,der)]
            return out
    
class Softmax():
    def __init__(self,inputD):
        self.inputD = inputD
        self.outputD = inputD

    def Pass(self,Input):
        self.input = Input

        ##perform a forward Pass
        e = 2.718281828459045235# eulers number

        # get the total sum
        self.sum = 0

        for I in self.input:
            self.sum += e**I

        # calculate the output for each neuron
        self.output = []

        for I in self.input:
            self.output.append((e**I)/self.sum)# calculate the output value
                    
    def Back(self,dLdO):
        self.dLdI = [dLdO[i]*self.output[i]*(1-self.output[i]) for i in range(self.outputD)]
        
        ## calculte dLdI for i != j
        for i in range(self.outputD):# for every xi
            for j in range(self.outputD):# for every output
                if i != j:
                    ## dOj/dIi x dL/dOj = dL/dIi
                    self.dLdI[i] -= (self.output[i]*self.output[j])*dLdO[j]

class CategoricalCrossentropy():
    def __init__(self,inputD):
        self.inputD = inputD
        self.loss = 0

    def Pass(self,input,y):
        self.input = input
        self.dLdI = []
        self.y = y# store the categories as well

        for I in range(self.inputD):
            self.loss -= log(self.input[I])*y[I]
            self.dLdI.append(0)

            if y[I] == 1:
                self.dLdI[I] = -1/self.input[I]
