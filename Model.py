## do 2D convolution in a 2D image and a 2D kernel

def Conv(image,kernel,bias):
    imD = [len(image[0]),len(image)]# width and height of the input image
    keD = [len(kernel[0]),len(kernel)]# width and height of the kernel
    outD = [1+imD[0]-keD[0],1+imD[1]-keD[1]]# width and height of the output

    output = []# initialize output matrix

    # loop though every output value in 
    for y in range(outD[1]):
        row = []# buffer for the output rows
        for x in range(outD[0]):
            o = bias# buffer for the output value

            # loop though all the values that take part in convolution
            for ky in range(keD[1]):
                for kx in range(keD[0]):
                    o += image[ky+y][kx+x]*kernel[ky][kx]# add the value to the convolution output

            row.append(o)
        output.append(row)

    return output

image = [[1,2,3],
         [4,5,6],
         [7,8,9]]

kernel = [[1,2],
          [3,4]]

bias = 1

output = Conv(image,kernel,bias)

for row in output:
    print(*row)