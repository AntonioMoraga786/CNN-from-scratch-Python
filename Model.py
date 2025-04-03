import random
import time
from math import log
import json
from multiprocessing import Manager,Pool

class Conv():
    def __init__(self,kD,n):
        self.n = n# number of neurons
        self.k = kD# save the kernel dimensions
        self.type = 0# id in the layer list (Conv layer Id)

    def init(self,inputD):
        # function to intialize all the paramers of the layers
        # kD is a variablel with the dimensions of the kernel [x,y,z]

        self.inputD = inputD# input image dimensions [x,y,z]
        self.outputD = [1+self.inputD[0]-self.k[0],1+self.inputD[1]-self.k[1],self.n]
        
        self.bias = [2*random.random()-1 for i in range(self.n)]

        ## initialize the weights
        
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

    def Back(self,dLdO,shared,batch,pos):# perform back propagation
        ## initialize derivative lists

        ## calculate dL/dB:

        # for the dLdO of every kernel (2D)
        n = 0
        for kernel in dLdO:# for every neuron
            dLdB = 0# buffer for dL/dB for a kernel
            for row in kernel:# for the row in each plane
                dLdB += sum(row)# add the sum of all the values in the row

            # add values to shared list
            shared[0][pos][n] += dLdB/batch
            n += 1

        ## v2
        self.dLdI = [[[0 for i in range(self.inputD[0])] for i in range(self.inputD[1])] for i in range(self.inputD[2])]
        
        # loop though all the output values
        for z in range(self.outputD[2]):# for every neuron
            for y in range(self.outputD[1]):# for every row in each output
                for x in range(self.outputD[0]):

                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                self.dLdI[kz][y+ky][x+kx] += self.kernel[z][kz][ky][kx]*dLdO[z][y][x]

                                val = self.input[kz][y+ky][x+kx]*dLdO[z][y][x]
                                shared[1][pos][z][kz][ky][kx] += val/batch# add value to shared derivatives list

    def Der(self):
        ## return a list with list with
        ## bias derivatives but full of 0s
        ## and the same for weights

        ## generate bias der
        bias = [0 for i in range(self.n)]

        ## generate weights der
        weights = [[[[0 for i in range(self.k[0])] for i in range(self.k[1])] for i in range(self.k[2])] for i in range(self.n)]

        return [bias,weights]

class dense():
    def __init__(self,n,I):
        self.type = 0
        self.kD = False
        self.inputD = I# input dimension (a singular integer value)
        self.bias = [2*random.random()-1 for i in range(n)]
        self.outputD = n
        self.n = n

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

    def Der(self):
        ## add the der function for the dense layer
        bias = [0 for i in range(self.n)]
        weights = [[0 for i in range(self.inputD)] for i in range(self.n)]

        return [bias,weights]

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
        self.type = 0

    def Pass(self,input,y):
        self.input = input
        self.dLdI = []
        self.y = y# store the categories as well

        for I in range(self.inputD):
            self.loss -= log(self.input[I])*y[I]
            self.dLdI.append(0)

            if y[I] == 1:
                self.dLdI[I] = -1/self.input[I]

class mBGD():
    def __init__(self):
        self.model = []

class Model():
    def __init__(self):
        # set default parameters
        self.lr = 1# set default learning Rate
        self.optimizer = False# store the optimizer object here
        self.loss = False# store the loss object here
        self.model = []# store the layer objects here
        self.minibatch = 1# mini-batch size (number of processes as well)
        self.inputD = False# store the input dimensions for the model
        self.epoch = 100# default epoch number

    def Add(self,layer):
        self.model.append(layer)

    def Pass(self,Input,Class,derivatives):
        ## function to perform a full Pass (forward and back) over the model with a specific input data
        ## model is already initialized
        ## function should add derivatives from this input into derivatives list
        ## derivatives list is a shared list between processes

        # start forward Pass
        self.input = Input# store the input value
        self.model[0].Pass(self.input)# forward pass on input layer

        ## forward pass on every other layer
        for i in range(len(self.model)-1):
            self.model[i+1].Pass(self.model[i].output)# pass the output from last function as input

        ## perform a Pass on the loss function
        self.loss.Pass(self.model[-1].output)

        ## perform the backward Passes
        dLdO = self.loss.dLdI# get initial derivative
        n = len(self.model)# number of layers in the model

        for i in range(n):
            # send, for backpropagation, dLdO, shared list, minibatch size, and layer in list
            self.model[n-i-1].Back(dLdO,derivatives,self.minibatch,n-i-1)# perform a backward Pass for every layer in the model
            dLdO = self.model[n-i-1].dLdI# get the dL/dI of the layer

        ## all the derivatives have been added into the shared list

    def Train(self,data,categories):
        ## initialize the model layers
        self.model[-1].init(self.inputD)# init initial layer
        
        # initialize the other layers
        for l in range(len(self.model)-1):
            self.model[l+1].init(self.model[l].outputD)# initialize each layer with the output from last

        # initialize

        ## start training the model
        for epoch in range(self.epoch):# loop for the desired number of epochs
            self.batches = len(data)//self.minibatch# get the number of minibatches in the whole dataset

            for n in range(self.batches):
                # get the data for this batch
                self.batch = data[:self.minibatch]
                data = data[self.minibatch:]

                self.Cat = categories[:self.minibatch]
                categories = categories[self.minibatch:]

                # create shared list to store the deriatives
                # empty list with two sub lists, one for bias derivatives, other for weight
                shared = [[],[]]
                
                # loop through every layer
                for model in self.model:
                    ders = model.Der()# get the 0 tensor for [bias,weights]
                    shared[0].append(ders[0])# add empty bias derivative for layer
                    shared[1].append(ders[1])# add empty weights derivative for layer

                shared = Manager().list(shared)# turn list into a shared list for pool

                # create a new process for each input in minibatch
                with Pool(processes=self.minibatch) as pool:# create self.minibatch processes
                    pool.starmap(self.Pass,[(In,Cat,shared) for In,Cat in zip(self.batch,self.Cat)])

                # pass the value into the optimizer and update model
                self.optimizer.model = self.model# update the optimizer model and prepare for updating
                self.optimizer.Pass(shared)# pass the derivatives to the optimizer
                self.model = self.optimizer.model# update the global model

    def predict(self,In):
        self.model[0].Pass(In)# feed input values to the input layeer

        for i in range(len(self.model)-1):# loop through all the other layers in the model
            self.model[i+1].Pass(self.model[0].output)# feed output from last layer as inptu

        ## get the index of the maximum value of output
        prob = max(self.model[-1].output)# get the maximum output value
        pos = self.model[-1].output.index(prob)# get the index of the max value

        return pos# return the index (categorical value)
    
    def save(self,name):
        ## function to save the model into a json format
        with open(f"./{name}.json","w") as file:
            output = []
            ## save model data
            data = {
                "type": "model",
                "optimizer": self.optimizer.type,
                "loss": self.loss.type,
                "inputD": self.inputD,
                "lr": self.lr,
                "minbatch": self.minibatch,
                "epoch": self.epoch,
            }
            output.append(data)

            ## save the layer data
            for layer in self.model:
                # store the important layer data
                data = {
                    "type": "layer",# its layer, optimizer, or loss...
                    "id": layer.type,# layer type (conv,dense,relu ...)
                    "kernel": layer.kernel,# kernel values
                    "bias": layer.bias,# bias values
                    "inputD": layer.inputD,#layer input dimension
                    "outputD": layer.outputD,#layer output dimension
                    "neuron": layer.n,# number of neurons
                    "k": layer.k# kernel dimensions (only valid for conv layers)
                }

                # save the data as json in output file
                output.append(data)

            ## save output in output file
            j = json.dumps(output,indent=4)
            file.write(j)

    def load(self,name):
        ## load a model from a json file
        with open(f"./{name}.json","r") as file:
            data = json.load(file)# load all json data into 

        ## get model specific data
        model = data[0]

        # set constant values
        self.lr = model["lr"]
        self.minibatch = model["minibatch"]
        self.inputD = model["inputD"]
        self.epoch = model["epoch"]

        ## set optimizer and loss functions
        
        # set optimizer
        optimizers = [mBGD()]
        self.optimizer = optimizers[model["optimizer"]]# get the optimizer from the list of optimizers

        # set loss function
        loss = [CategoricalCrossentropy()]
        self.loss = loss[model["loss"]]

        ## add each layer
        for layer in data[1:]:
            layers = [Conv(layer["k"],layer["neuron"],layer["inputD"]), 
                      dense(layer["neuron"],layer["inputD"]), 
                      ReLU(layer["inputD"]), 
                      Softmax(layer["inputD"])]
            
            self.Add(layers[layer["id"]])# add layer into the model
            
            # update model parameters
            self.model[-1].kernel = layers["kernel"]# set model.kernel equal to saved value for kernel
            self.model[-1].bias = layers["bias"]# same for bias

            # layer added and updated, continue to add next layer