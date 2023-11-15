import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from tqdm.auto import tqdm


from torchvision import datasets, transforms

np.random

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)


dataiter = iter(trainloader)
images, labels = dataiter.__next__()


class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.w1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.w3 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b1 = np.random.randn(hidden_size) / np.sqrt(input_size)
        self.b2 = np.random.randn(hidden_size) / np.sqrt(hidden_size)
        self.b3 = np.random.randn(output_size) / np.sqrt(hidden_size)


    def relu(self, x):
        return np.maximum(0, x)
    def drelu(self, x):
        return np.where(x <= 0, 0, 1)
    
    def leaky_relu(self, x):
        return np.maximum(0.1*x, x)
    def dleaky_relu(self, x):
        return np.where(x <= 0, 0.1, 1)
    def tanh(self, x):
        return np.tanh(x)
    def dtanh(self, x):
        return 1 - np.power(np.tanh(x), 2)
    def softmax(self,x):
        z = x - max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator

        return softmax
    def dsoftmax(self, x):
        s = self.softmax(x)
        return np.diagflat(s) - np.dot(s, s.T)
    
    def one_hot_encode(self, y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot
    
    def categorical_cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]
    
    def binary_cross_entropy(self, y_true, y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Calculate the binary cross-entropy separately for each output neuron
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=0)
        return np.mean(loss)
    
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))
    
 
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2)+self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3)+self.b3
        self.a3 = self.z3
        return self.a3
    
    def backward(self, x, y, o, learning_rate):
        # Backpropagation
        y_one_hot = self.one_hot_encode(y, 10)  # Assuming 10 classes
        error = o - y_one_hot
        d_output = 2 * error / error.shape[0]  # Derivative of MSE w.r.t output
        #d_output = error * self.dsoftmax(self.z3)
        d_hidden_layer2 = np.dot(d_output, self.w3.T) * self.drelu(self.a2)
        d_hidden_layer = np.dot(d_hidden_layer2, self.w2.T) * self.drelu(self.a1)
        # Update weights
        self.w1 -= np.dot(x.T, d_hidden_layer) * learning_rate
        self.w2 -= np.dot(self.a1.T, d_hidden_layer2) * learning_rate
        self.w3 -= np.dot(self.a2.T, d_output) * learning_rate
        self.b1 -= np.sum(d_hidden_layer, axis=0) * learning_rate
        self.b2 -= np.sum(d_hidden_layer2, axis=0) * learning_rate
        self.b3 -= np.sum(d_output, axis=0) * learning_rate
        return self.mse(y_one_hot, o)
    
model = SimpleNN(784, 128, 10)


images, labels = next(iter(trainloader))
images.resize_(1, 784)



time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in tqdm(trainloader):
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        images=np.array(images)
        labels = np.array(labels)
        #print(images.shape)
        #print(labels.shape)
        labels=np.array(labels)
        output = model.forward(images)

        loss= model.backward(images, labels, output, 0.0001)
        #print(loss)
        running_loss += loss

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)


def predict(model, x):
    outputs = model.forward(x)
    # Choose the class with the highest score
    predicted_class = np.argmax(outputs, axis=1)
    return predicted_class

# Test the model
correct_count, all_count = 0, 0

images, labels = next(iter(valloader))
images.resize_(1, 784)
for images, labels in valloader:
    images = images.view(images.shape[0], -1)  # Flatten images
    images = np.array(images)
    labels = np.array(labels)

    predicted_labels = predict(model, images)
    correct_count += np.sum(predicted_labels == labels)
    all_count += labels.shape[0]

print("Accuracy: {:.2f}%".format(100 * correct_count / all_count))

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))
