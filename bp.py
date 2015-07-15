from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SigmoidLayer

net = buildNetwork(20,5,1,bias=True,hiddenclass=SigmoidLayer,outclass=SigmoidLayer)

ds = SupervisedDataSet(20,1)

f = open("german.dat","r")

for line in open("german.dat"):
	line = f.readline()
	data = line.split(",")
	input = []
	output = []
	for i in range(0,len(data)):
		if i < len(data) - 1:
			input.append(data[i])
		else:
			output.append(data[i])
	ds.addSample(input,output)



trainer = BackpropTrainer(net,ds,learningrate=0.5,momentum=0.1,verbose=True)

while 1:

	error = trainer.trainEpochs(epochs=50)

	if error < 0.01:

		break


print trainer.activate([1,30,2,2,6350,5,5,4,3,1,4,2,31,3,2,1,3,1,1,1])

