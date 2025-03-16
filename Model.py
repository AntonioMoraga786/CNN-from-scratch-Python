def Conv(image,kernel,bias):
    # same thing as V0 just with a extra dimension
    imD = [len(image[0][0]),len(image[0]),len(image)]# dimensions of the input image
    keD = [len(kernel[0][0]),len(kernel[0]),len(kernel)]# diemnsions of the kernel
    outD = [1 + imD[0] - keD[0],1 + imD[1] - keD[1],1 + imD[2] - keD[2]]# dimensions of the output

    output = []# initialize the output list

    # loop through all the output values
    for z in range(outD[2]):
        plane = []# buffer to store the matrix before adding it to output
        for y in range(outD[1]):
            row = []# buffer to store the rows before being appended into plane
            for x in range(outD[0]):
                o = bias# calculate the specific output value

                # loop though all the values in the convolution
                for kz in range(keD[2]):
                    for ky in range(keD[1]):
                        for kx in range(keD[0]):
                            o += image[z+kz][y+ky][x+kx]*kernel[kz][ky][kx]# add value to output

                row.append(o)
            plane.append(row)
        output.append(plane)

    return output


image = [[[1,2,3],
         [4,5,6],
         [7,8,9]],
         [[4,5,6],
         [1,2,3],
         [7,8,9]]]

kernel = [[[1,2],
          [3,4]]]

bias = 1

output = Conv(image,kernel,bias)

for plane in output:
    print("")
    for row in plane:
        print(*row)