import csv
import time
import matplotlib
import numpy
from scipy.special import expit

#Parameters
epochs = 50

#learining rate
eta = 0.1

#momentum
alpha = 0.9

#input layer weight matrix.
w1 = numpy.random.uniform(-0.05,0.05,(784,100))

#hidden layer weight matrix.
w2 = numpy.random.uniform(-0.05,0.05,(100,10))

#bias input.
biasInput = 1

#bias weights for hidden layer.
w1b = numpy.random.uniform(-0.05,0.05,(1,100))

#bias weights for output layer.
w2b = numpy.random.uniform(-0.05,0.05,(1,10))

#matrix to store previous hidden layer bias weights
itw1b = numpy.zeros((1,100))

#matrix to store previous output layer bias weights.
itw2b = numpy.zeros((1,10))

#matrix to store previous input-hidden layer weights.
itw1 = numpy.zeros((784,100))

#matrix to store previous hidden-output layer weights.
itw2 = numpy.zeros((100,10))

inputs = numpy.genfromtxt('./csv/mnist_train.csv',delimiter=',')
test = numpy.genfromtxt('./csv/mnist_test.csv',delimiter=',')
classes = numpy.array([0,1,2,3,4,5,6,7,8,9])

'''Return the sigmoid of n.'''
def sigmoid(n):
    return expit(n)

#end

'''Feed forward, accept the input,corresponding labels and data set name - training or testing.'''
def feedForward(e,inputs,labels,setName):
    global w1,w2,biasInput,w1b,w2b,itw1b,itw1,itw2b,itw2
    actual = []
    predictions = []
    for i in range(inputs.shape[0]):
        actual.append(labels[i])
        inputData = numpy.reshape(inputs[i],(1,inputs[i].shape[0]))
        l1 = biasInput*w1b
        z1 = numpy.dot(inputData,w1) + l1
        a1 = sigmoid(z1)
        l2 = biasInput*w2b
        z2 = numpy.dot(a1,w2) + l2
        a2 = sigmoid(z2)
        prediction = numpy.argmax(a2)
        predictions.append(prediction)
        if(setName == 'training' and e != 0):
            itw1b,itw1,itw2b,itw2 = learn(e,inputData,a1,a2,prediction,labels[i])
    #end
    generateConfusionMatrix(e,classes,numpy.array(actual),numpy.array(predictions),setName,itw1b.shape[1])
    time.sleep(2)
#end


'''Perform back-propagation with inputs as training example,a1 as activations in hidden layer,
    a2 being activations of output layer,prediction being class predicted for input example and label 
    being actual label of the class.'''
def learn(epochs,inputs,a1,a2,prediction,label):
    global biasInput,w2,w2b,eta,alpha,itw2b,itw2
    hotVector = numpy.insert((numpy.zeros((1,9))+0.1),label,0.9)
    errors2 = a2 * (1 - a2) * (hotVector - a2)
    errors1 = a1 * (1 - a1) * numpy.dot(errors2,numpy.transpose(w2))
    oldw2 = updateWeightsH(epochs,a1,errors2)
    oldw1 = updateWeightsI(epochs,inputs,errors1)
    return oldw1[0],oldw1[1],oldw2[0],oldw2[1]
#end


'''Back-propagate - update the hidden-output layer weights with output layer errors and activations a1 in hidden layer. '''
def updateWeightsH(epochs,a1,errors):
    global biasInput,w2b,w2,eta,alpha,itw2,itw2b
    deltaw2b = (eta * numpy.dot(errors,biasInput)) + (alpha * itw2b)
    w2b = w2b + deltaw2b
    deltaw2 = (eta * numpy.dot(numpy.transpose(a1),errors)) + (alpha * itw2)
    w2 = w2 + deltaw2
    return deltaw2b,deltaw2
#end


'''Back-propagate - update input to hidden layer weights with errors as errors in hidden layer. '''
def updateWeightsI(epochs,inputs,errors):
    global biasInput,w1b,w1,eta,alpha,itw1b,itw1
    deltaw1b = (eta * numpy.dot(errors,biasInput)) + (alpha * itw1b) 
    w1b = w1b + deltaw1b
    deltaw1 = (eta * numpy.dot(numpy.transpose(inputs),errors)) + (alpha * itw1) 
    w1 = w1 + deltaw1
    return deltaw1b,deltaw1
#end


'''Separate training labels from training data.'''
def grabLabels(inputs):
    labels = numpy.zeros(inputs.shape[0],numpy.int32)
    for i in range(inputs.shape[0]):
        labels[i] = inputs[i,0]
    #end
    return labels

#end

''' Scale the input.'''
def preProcessInput(inputs):
    for i in range(inputs.shape[0]):
        inputs[i,:] = inputs[i,:]/255.0
    #end
    return inputs[:,1:]
#end


'''Generate confusion matrix after each epoch of training.'''
def generateConfusionMatrix(epoch,classes,actual,predictions,setName,n):
    global eta
    confusionMatrix = numpy.zeros((len(classes),len(classes)),numpy.int32)
    for a,p in zip(actual,predictions):
        confusionMatrix[a][p] = confusionMatrix[a][p] + 1
    #end
    printConfusionMatrix(confusionMatrix)
    accuracy = (actual == predictions).sum()/float(len(actual))*100
    print(setName+" - accuracy:",accuracy)
    writeData(eta,epoch,accuracy,setName,n)    
#end


'''After each epoch write data to csv file to generate Accuracy vs Epochs plot.'''
def writeData(eta,epoch,accuracy,setName,n):
    global alpha,inputs
    f = open('exp-3-'+setName+'-data-size-'+str(inputs.shape[0])+'-.csv', 'a',newline='')
    with f:
        writer = csv.writer(f)
        writer.writerow([epoch,accuracy])
#end


'''Helper to print the confusion matrix.'''
def printConfusionMatrix(confusionMatrix):
    print("confusion matrix\n")
    for i in range(confusionMatrix.shape[0]):
        print(confusionMatrix[i])
    #end


'''Start the process.'''
def start():
    global epochs,inputs,test,testData,itw1b,itw1,itw2b,itw2
    labels = grabLabels(inputs)
    inputs = preProcessInput(inputs)
    testLabels = grabLabels(test)
    testData = preProcessInput(test)
    # Uncomment to run with permutation experiment.
    # randomS = numpy.random.permutation(labels.shape[0])
    # random1S = randomS[0:int(inputs.shape[0]/4)]
    # inputs,labels = inputs[random1S],labels[random1S]
    for e in range(epochs):
        print("\nepoch : ",e)
        feedForward(e,inputs,labels,'training')
        feedForward(e,testData,testLabels,'testing')
    #end
numpy.random.seed(None)
start()
