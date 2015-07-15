from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer
from pybrain.structure import SigmoidLayer

net = buildNetwork(2,10,1,bias=True,hiddenclass=SigmoidLayer)

ds = SupervisedDataSet(2,1)

ds.addSample([0.5,0.5],[1])
ds.addSample([0.2,0.5],[1])
ds.addSample([-0.3,-0.3],[-1])
ds.addSample([-0.1,-1.0],[-1])

trainer = BackpropTrainer(net,ds,learningrate=0.1,momentum=0.1,verbose=True)

while 1:

	error = trainer.trainEpochs(epochs=100)

	if error < 0.01:

		break


print net.activate([1.4,1.2])
