from Model import dense
from Model import ReLU,Softmax
from Model import CategoricalCrossentropy
from Model import mBGD,Model
import random
from nnfs.datasets import spiral_data

classes = 4# number of classes
# get data 
INPUT,Preds = spiral_data(samples=400,classes=classes)

# get the data in the correct format
data = []
for item in zip(INPUT,Preds):
    ## one Hot encode categories
    CAT = [0 for i in range(classes)]
    CAT[item[1]] = 1

    data.append([item[0],CAT])

# shuffle the data so that it leads to more stable training
random.shuffle(data)

# split into 80% Training Data and 20% Test
TrainingData = data[:320]
TestData = data[320:]


model = Model()

## input layer
model.Add(dense(50))
model.Add(ReLU())

model.Add(dense(50))
model.Add(ReLU())

model.Add(dense(50))
model.Add(ReLU())

## output layer
model.Add(dense(classes))
model.Add(Softmax())

model.loss = CategoricalCrossentropy(classes)
model.optimizer = mBGD()

model.inputD = 2
model.epoch = 300
model.minibatch=20# data samples per batch
model.lr = 0.1
model.decay = 0.1/600

In = []
Out = []

for item in TrainingData:
    In.append(item[0])
    Out.append(item[1])

model.Train(In,Out)