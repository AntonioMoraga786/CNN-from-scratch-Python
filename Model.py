import random

class Conv():
    def __init__(self,kD):
        # function to intialize all the paramers of the layers
        # kD is a variablel with the dimensions of the kernel [x,y,z]

        ## initialize the bias between -1 and 1
        self.bias = 2*random.random()-1

        ## initialize the weights
        self.k = kD# save the kernel dimensions
        self.kernel = []# initialize the kernel weights list

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
            self.kernel.append(plane)
        
        ## weight initalization is done.

    def Pass(self,image):
        # perform a forward pass
        self.input = image# store the input matrix as well

        # calculate all of the dimensions
        self.im = [len(image[0][0]),len(image[0]),len(image)]
        self.outputD = [1+self.im[0]-self.k[0],1+self.im[1]-self.k[1],1+self.im[2]-self.k[2]]

        # initialize output list
        self.output = []

        # loop though all the output values
        for z in range(self.outputD[2]):
            plane = []# buffer for output vals
            for y in range(self.outputD[1]):
                row = []# buffer for output vals

                for x in range(self.outputD[0]):
                    o = self.bias

                    # loop though allthe values in convolution of this value
                    for kz in range(self.k[2]):
                        for ky in range(self.k[1]):
                            for kx in range(self.k[0]):
                                o += self.input[z+kz][y+ky][x+kx]*self.kernel[kz][ky][kx]

                    row.append(o)
                plane.append(row)
            self.output.append(plane)

        # Convolution is done

# define the input image
image = [[[1,2,3],
         [4,5,6],
         [7,8,9]],
         [[1,2,3],
         [4,5,6],
         [7,8,9]]]

# initialize the kernel as a 2x2x2 shape and make a forward pass
conv = Conv([2,2,2])
conv.Pass(image)

# print the outputs
print("kernel")
for plane in conv.kernel:
    print("")
    for row in plane:
        print(*row)

print("")
print("output")

for plane in conv.output:
    print("")

    for row in plane:
        print(*row)
