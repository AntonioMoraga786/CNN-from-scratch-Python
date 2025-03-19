import random
import time

class Conv():
    def __init__(self,kD,n):
        # function to intialize all the paramers of the layers
        # kD is a variablel with the dimensions of the kernel [x,y,z]
        self.n = n# number of neurons
        
        self.bias = []
        ## initialize the bias between -1 and 1 for all neurons
        for i in range(self.n):
            self.bias.append(2*random.random()-1)

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
            
        ## weight initalization is done.

    def Pass(self,image):
        # perform a forward pass
        self.input = image# store the input matrix as well

        # calculate all of the dimensions
        self.im = [len(image[0][0]),len(image[0]),len(image)]
        self.outputD = [1+self.im[0]-self.k[0],1+self.im[1]-self.k[1],self.n]

        # initialize output list
        self.output = []

        # loop though all the output values
        for n in range(self.n):
            plane = []# buffer for output vals

            for y in range(self.outputD[1]):
                row = []# buffer for output vals

                for x in range(self.outputD[0]):
                    o = self.bias[n]

                    # loop though allthe values in convolution of this value
                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                o += self.input[kz][y+ky][x+kx]*self.kernel[n][kz][ky][kx]

                    row.append(o)
                plane.append(row)
            self.output.append(plane)

        # Convolution is done

    def Back(self,dLdO):# perform back propagation
        t1 = time.time()
        ## initialize derivative lists
        self.dLdI = []#dL/dI
        self.dLdW = []#dL/dW
        self.dLdB = []#dL/dB

        t2 = time.time()
        ## calculate dL/dB:

        # for the dLdO of every kernel (2D)
        for kernel in dLdO:
            dLdB = 0# buffer for dL/dB for a kernel
            for row in kernel:# for the row in each plane
                dLdB += sum(row)# add the sum of all the values in the row

            self.dLdB.append(dLdB)# add value into bias derivatives

        t4 = time.time()
        ## v2
        self.dLdI = [[[0 for i in range(self.im[0])] for i in range(self.im[1])] for i in range(self.im[2])]
        self.dLdW = [[[[0 for i in range(self.k[0])] for i in range(self.k[1])] for i in range(self.k[2])] for i in range(self.n)]
        
        # loop though all the output values
        for z in range(self.outputD[2]):# for every neuron
            for y in range(self.outputD[1]):
                for x in range(self.outputD[0]):

                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                self.dLdI[kz][y+ky][x+kx] += self.kernel[z][kz][ky][kx]*dLdO[z][y][x]
                                self.dLdW[z][kz][ky][kx] += self.input[kz][y+ky][x+kx]*dLdO[z][y][x]

        t5 = time.time()
        print("OLD:",t4-t3)
        print("NEW:",t5-t4)
        print("Full OLD:",t4-t1)
        print("Full NEW:",t5-t1+t3-t4)


# define the input image
image = [[[1,2,3],
         [4,5,6],
         [7,8,9]],
         [[1,2,3],
         [4,5,6],
         [7,8,9]]]

dLs = [[[1,1],
        [1,1]],
        [[1,1],
        [1,1]]]

# initialize the kernel as a 2x2x2 shape and make a forward pass
t1 = time.time()
conv = Conv([2,2,2],2)
conv.Pass(image)
conv.Back(dLs)

t2 = time.time()
print(t2-t1)

# print the outputs
print("kernel")
for neuron in conv.kernel:
    print("")
    print("")

    for plane in neuron:
        print("")

        for row in plane:
            print(*row)

print("")
print("output")

for plane in conv.output:
    print("")

    for row in plane:
        print(*row)

print("")
print("dL/dI")

for plane in conv.dLdI:
    print("")
    for row in plane:
        print(*row)

print("")
print("dL/dW")

i = 0
for neuron in conv.dLdW:
    print(f"Neuron {i}")
    print("")
    i += 1

    for plane in neuron:
        print("")
        for row in neuron:
            print(*row)

print("")
print("dL/dB")

print(*conv.dLdB)