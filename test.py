import time
## test the convolutional layer
"""
from Model import Conv

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
""""
from Model import dense

input = [1,2,3,4,5]
dLs = [1,1]

layer = dense(2,5)
layer.Pass(input)
layer.Back(dLs)

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

print("")
print("")
print("dL/dO")

print(*layer.dLdB)


print("")
print("")
print("dL/dI")

print(*layer.dLdI)

print("")
print("")
print("dL/dW")

i = -1
for neuron in layer.dLdW:
    i += 1
    print(f"neuron {i}")
    print("")

    print(*neuron)
"""
##Test the ReLU Activation
"""
from Model import ReLU

In = [[[-1,-3,-5,-7,-9,2,4,6,8,0],
      [-1,-3,-5,-7,-9,2,4,6,8,0]],
      [[-1,-3,-5,-7,-9,2,4,6,8,0],
      [-1,-3,-5,-7,-9,2,4,6,8,0]]]

dLs = [[[1,1,1,1,1,1,1,1,1,1],
       [1,1,1,1,1,1,1,1,1,1]],
       [[1,1,1,1,1,1,1,1,1,1],
       [1,1,1,1,1,1,1,1,1,1]]]

t1 = time.time()
layer = ReLU([10,2,2])
t2 = time.time()
layer.Pass(In)
t3 = time.time()
layer.Back(dLs)

t4 = time.time()

print("INIT",t2-t1)
print("PASS",t3-t2)
print("BACK",t4-t3)

print("")
print("OUTPUT")
for row in layer.output:
    print(*row)

print("")
print("dL/dI")
for row in layer.dLdI:
    print(*row)
"""
## Test the Softmax activation function
"""
from Model import Softmax

input = [1,2,3,4,5]

dLs = [0,0,0,0,1]

layer = Softmax(5)
layer.Pass(input)
layer.Back(dLs)

print("output")
print(*layer.output)

print("")
print("dL/dI")
print(*layer.dLdI)
"""
## Test The loss function
"""
from Model import CategoricalCrossentropy

input = [0.1,.1,.2,.6]
y = [0,0,0,1]# one hot encoded category

loss = CategoricalCrossentropy(4)
loss.Pass(input,y)

print("loss")
print(loss.loss)

print("dL/dI")
print(*loss.dLdI)
"""

## test Model class

from Model import Model,dense,CategoricalCrossentropy

model = Model()
model.Add(dense(10,10))
model.Add(dense(10,10))
model.Add(dense(10,10))

model.loss = CategoricalCrossentropy(10)
model.optimizer = CategoricalCrossentropy(10)

model.save("file")