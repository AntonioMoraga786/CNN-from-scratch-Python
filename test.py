"""
## test the convolutional layer

from Model import Conv
import time
# define the input image
image = [[[1,2,3],
         [4,5,6],
         [7,8,9]],
         [[1,2,3],
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
conv = Conv([3,3,3],2)
conv.Pass(image)
conv.Back(dLs)

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

"""
## test the dense layer

from Model import dense

input = [1,2,3,4,5]

layer = dense(2,5)
layer.Pass(input)

print("")
print("bias")
print(*layer.bias)

print("")
print("")
print("kernel")
i = -1

for neuron in layer.kernel:
    i += 1
    print("")
    print(f"neuron {i}")

    print(*neuron)

print("")
print("output")
print(*layer.output)