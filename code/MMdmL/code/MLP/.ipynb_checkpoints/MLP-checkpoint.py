import numpy as np
import abc

class BaseActivationFunction(metaclass=abc.ABCMeta):
    """ This is the base class for all activation functions """  
    def __init__(self, type):
        self.type = type

        # this function keeps the current value as well as derivative
        self.value = 0
        self.derivative = 0
            
    @abc.abstractmethod    
    def getValue(self, input):
        pass
        
    def getDerivative(self):
        return self.derivative 
        
    def getType(self):
        return self.type    
        
class SigmoidActivationFunction(BaseActivationFunction):
    """ This is the sigmoid activation function """    
    def __init__(self):
        super().__init__('Sigmoid')
    
    def getValue(self, input):
        self.value      = 1.0/(1.0+np.exp(-input))
        
        # value of derivative is updated immediately
        self.derivative = np.exp(-input)/(1+np.exp(-input))**2
        
        return self.value   
        
class ReLUActivationFunction(BaseActivationFunction):
    """ This is the ReLU activation function """    
    def __init__(self):
        super().__init__('ReLU')
    
    def getValue(self, input):
        self.value = np.maximum(0,input)
        
        # value of derivative is updated immediately
        if (input>0.0):
            self.derivative = 1
        else:
            self.derivative = 0.001    
        
        return self.value     
        
class LinearActivationFunction(BaseActivationFunction):
    """ This is the Linear activation function """    
    def __init__(self):
        super().__init__('Linear')
    
    def getValue(self, input):
        self.value = input
        
        # value of derivative is updated immediately
        self.derivative = 1    
        
        return self.value              
   
class TanhActivationFunction(BaseActivationFunction):
    """ This is the Linear activation function """    
    def __init__(self):
        super().__init__('Tanh')
    
    def getValue(self, input):
        self.value = np.tanh(input)
        
        # value of derivative is updated immediately
        self.derivative = 1.0/np.cosh(input)**2    
        
        return self.value 
        
class BaseLossFunction(metaclass=abc.ABCMeta):    
    """ This is the base class for all Lossfunctions """
    def __init__(self,type, outputLayer):
        self.outputLayer = outputLayer
        self.type = type
        self.value = 0
        self.derivatives = np.empty(outputLayer.numberOfPerceptrons())

    def getValue(self):
        return self.value

    @abc.abstractmethod
    def getDerivative(self,index):
        pass
        
    @abc.abstractmethod
    def update(self,output):
        pass             
                               
class MeanSquaredLossFunction(BaseLossFunction):
    """ This is the mean squared loss function """
    def __init__(self, outputLayer):
        super().__init__('MeanSquaredLossFunction', outputLayer)
        
    def update(self,output):
        val = 0
        num = output.shape[0]
        for i in range(num):
            outmlp = self.outputLayer.getPerceptron(i).getValue()
            val += (outmlp-output[i])**2
            self.derivatives[i] = 2.0/num*(outmlp-output[i])
        
        self.value = 1.0/num * val    
        
    def getDerivative(self,index):
        return self.derivatives[index]           
        
class CrossEntropyLossFunction(BaseLossFunction):
    """ This is the cross entropy loss function """
    def __init__(self, outputLayer):
        super().__init__('CrossEntropyLossFunction', outputLayer)
    
    def update(self, output):
        val = 0
        den = 0
        num = output.shape[0]
        
        for i in range(num):
            outmlp = self.outputLayer.getPerceptron(i).getValue()
            den += np.exp(outmlp)
        
        for j in range(num): 
            outmlp = self.outputLayer.getPerceptron(j).getValue()            
            val += -output[j]*(outmlp-np.log(den))
            self.derivatives[j] = np.exp(outmlp)/den - output[j]
    
        
        self.value = val    
        
    def getDerivative(self,index):
        return self.derivatives[index]                                                         
                                            
class Perceptron:
    """ This class encodes a single perceptron """
    def __init__(self, mlp, layer, num_inputs=0, activationFunction='sigmoid', isInputPerceptron=False):
        self.mlp = mlp
        self.num_inputs = num_inputs
        self.layer = layer
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand()            
        self.derivatives = np.empty(num_inputs)
        self.isInputPerceptron = isInputPerceptron
        self.value = 0
        self.da = {'sigmoid': SigmoidActivationFunction, 'relu': ReLUActivationFunction, 'linear': LinearActivationFunction, 'tanh': TanhActivationFunction}
        self.activationFunction=self.da[activationFunction]()
        
    def update(self,precedingLayer, index=None):
        # create new input by multiplying weights with predecessors' corresponding output
        val=0
                                        
        for i in range(0,self.num_inputs):
            val += self.weights[i]*precedingLayer.getPerceptron(i).getValue()
        
        # the activation function is supposed to update it's derivative within function getValue()
        self.value = self.activationFunction.getValue(val+self.bias)
        
        # update derivatives
        self.derivatives = self.activationFunction.getDerivative()*self.weights
                
        
    def backprop(self, precedingLayer, succeedingLayer, index, learningRate):
        # start updating weights and biases
        val=0
        
        if (precedingLayer==None):
            # this is obviously the input layer
            return
        
        elif (succeedingLayer==None):
            # this perceptron's layer is obviously the output layer
            val = self.mlp.getLossFunction().getDerivative(index)
            
            # the result is stored as perceptron's current value
            self.value = val                        

            val = val * self.activationFunction.getDerivative()
        
            for j in range(0,self.num_inputs):
                self.weights[j] -= val*precedingLayer.getPerceptron(j).getValue()*learningRate
            
            self.bias -= val*learningRate
        else:    
            for i in range(0,succeedingLayer.numberOfPerceptrons()):
                val += succeedingLayer.getPerceptron(i).getValue()*succeedingLayer.getPerceptron(i).getDerivative(index)
            
            # the result is stored as perceptron's current value
            self.value = val    
        
            val = val * self.activationFunction.getDerivative()
            
            for j in range(0,self.num_inputs):
                self.weights[j] -= val*precedingLayer.getPerceptron(j).getValue()*learningRate
            
            self.bias -= val*learningRate  
        
    def getValue(self):
        return self.value    
        
    def setValue(self,val):
        self.value = val    
        
    def getDerivative(self,index):
        return self.derivatives[index]    
        
    def getWeights(self):
        return self.weights
        
    def getBias(self):
        return self.bias        
        
    def setActivationFunction(self, activationFunction):
        self.activationFunction=self.da[activationFunction]()    
        
        
class Layer:
    """ This class encodes a single layer """
    
    def __init__(self, mlp, num_perceptrons, activationFunction, precedingLayer):
        self.mlp = mlp
        self.numPerceptrons = num_perceptrons
        self.precedingLayer = precedingLayer
        self.perceptrons = np.empty(num_perceptrons,dtype=Perceptron)
        self.isInputLayer = None
                        
        if (precedingLayer==None):
            isInputLayer=True                        

            for i in range(num_perceptrons):
                self.perceptrons[i] = Perceptron(mlp, layer=self, isInputPerceptron=True)
            
        else:     
            isInputLayer=False   
            for i in range(num_perceptrons):
                self.perceptrons[i] = Perceptron(mlp, layer=self, num_inputs=precedingLayer.numberOfPerceptrons(), activationFunction=activationFunction)
    
    def numberOfPerceptrons(self):
        return self.numPerceptrons
        
    def update(self):
        for i in range(self.numPerceptrons):
            self.perceptrons[i].update(self.precedingLayer, i)
            
    def backprop(self,succeedingLayer,learningRate):
        if (succeedingLayer==None):
            # this layer is actually the output layer
            for i in range(self.numPerceptrons):
                self.getPerceptron(i).backprop(precedingLayer=self.precedingLayer, succeedingLayer=None, index=i, learningRate=learningRate)
        else:
            # this is obviously a hidden layer
            for i in range(self.numPerceptrons):
                self.getPerceptron(i).backprop(precedingLayer=self.precedingLayer, succeedingLayer=succeedingLayer, index=i, learningRate=learningRate)                       
        
    def getPerceptron(self,index):
        return self.perceptrons[index]
        
    def setActivationFunction(self, activationFunction):
        for i in range(self.numberOfPerceptrons()):
            self.perceptrons[i].setActivationFunction(activationFunction)    
                
            
class MultiLayerPerceptron:
    """ This class encodes the actual multilayer perceptron """
    
    def __init__(self, topology, lossFunction, activationFunction='sigmoid'):
        self.topology = topology
        self.layers = np.empty(topology.size,dtype=Layer)
        self.numLayers = topology.size
        self.dl = {'meansquared': MeanSquaredLossFunction, 'crossentropy': CrossEntropyLossFunction}
                
        self.layers[0] = Layer(self, self.topology[0], activationFunction, precedingLayer=None)
        for i in range(1, topology.size):
            self.layers[i] = Layer(self, self.topology[i], activationFunction, precedingLayer=self.layers[i-1])
        
        self.lossFunction = self.dl[lossFunction](self.layers[self.numberOfLayers()-1])
 
    #########################
    # Some helper functions # 
    ######################### 
    def numberOfLayers(self):
        return self.numLayers        
        
    def getLayer(self,index):
        return self.layers[index]        
        
    def setValuesOfInputLayer(self,input):
        # check whether the input fits the number of input perceptrons
        if (input.size != self.layers[0].numberOfPerceptrons()):
            print('size of data does not fit')
            return False
        else:
            # fill in values of input perceptrons
            for i in range(self.layers[0].numberOfPerceptrons()):
                self.layers[0].getPerceptron(i).setValue(input[i])        
            
            return True    
            
    def getLossFunction(self):
        return self.lossFunction      

    ########################
    # show MLPs structure  # 
    ########################
    def summary(self):
        print('---------------------------------')
        print('Structure: ')
        for i in range(self.numberOfLayers()): 
            print('Layer {}:'.format(i))
            print('  Number of Neurons: {}'.format(self.layers[i].numberOfPerceptrons()))
            print('  Activation Function: {}'.format(self.layers[i].getPerceptron(0).activationFunction))   
            print()
            
        print('Loss Function: {}'.format(self.lossFunction))    
        print('---------------------------------')    

    ########################################################    
    # features are supposed to be stored in 1d numpy array #
    ########################################################                                       
    def predict(self,input, softmax=False):
        # check whether the input fits the number of input perceptrons
        if (self.setValuesOfInputLayer(input)):
            for j in range(1, self.numberOfLayers()):
                self.getLayer(j).update()
            val = 0    
            output = np.empty(self.getLayer(self.numberOfLayers()-1).numberOfPerceptrons())
            for k in range(output.size):
                output[k] = self.getLayer(self.numberOfLayers()-1).getPerceptron(k).getValue()
                
            if(softmax):
                den=0
                for i in range(output.size):    
                    den += np.exp(output[i])
                    output[i]=np.exp(output[i])
                
                output = 1.0/den * output                        
                
            return output                       
        return None   
        
    ################################################################    
    # features-vectors are supposed to be stored in 2d numpy array #
    # number of samples x number of features                       #
    ################################################################       
    def predictAll(self,inputs,softmax=False):
        # check if dimensions match
        if (inputs.shape[1] != self.layers[0].numberOfPerceptrons()):
            print('size of data does not fit')
            return
        
        results = np.empty((inputs.shape[0], self.layers[self.numberOfLayers()-1].numberOfPerceptrons()))
        for i in range(inputs.shape[0]):
            results[i,:] = self.predict(inputs[i], softmax)
            
        return results                        
        
    ################################################################    
    # input data is supposed to be stored in 2d numpy array        #
    # number of samples x number of input neurons                  #
    ################################################################   
    # input data is supposed to be stored as (numberOfSample, sizeOfInputLayer)    
    def learn(self,inputs, outputs, learningRate, numberOfEpochs, output_epochs=1):
        # check whether the input and out fits the number of input and output perceptrons
        if (inputs.shape[1] != self.layers[0].numberOfPerceptrons() or outputs.shape[1] != self.layers[self.numberOfLayers()-1].numberOfPerceptrons() or inputs.shape[0] != outputs.shape[0]):
            print('size of data does not fit')
            return
        else:
            for i in range(numberOfEpochs):
                err = 0
                
                # shuffle data first inputs and outputs
                p= np.random.permutation(outputs.shape[0])
                inputs  = inputs[p]
                outputs = outputs[p] 
                for k in range(inputs.shape[0]):
                    self.predict(inputs[k])
                    self.getLossFunction().update(outputs[k])
                    err += self.getLossFunction().getValue()
                    
                    # loop over all but the first layer in reverse order
                    self.getLayer(self.numberOfLayers()-1).backprop(None, learningRate)
                    for l in range(self.numberOfLayers()-2,0,-1):
                        self.getLayer(l).backprop(self.getLayer(l+1),learningRate)
                        
                if (i%output_epochs == 0):
                    print("Mean error in epoch {} : {}".format(i, err/inputs.shape[0]))        
            print("training has been finished ...")    
                    
                    
                
                    
                
        
           
            
                
		
	