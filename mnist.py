import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

#Download the MNIST dataset
#We have a test set and a training set
trainset = datasets.MNIST('./data', download=True, train=True, transform=transform)
valset = datasets.MNIST('./data', download=True, train=False, transform=transform)

#Loads the data in a batch of 64 images
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

#Get a batch from the dataset and print its outputs and shape
images, labels = next(iter(trainloader))

print(images.shape)
print(labels.shape)


#Display one image
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')
#plt.show()

#Display more of the images
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
#plt.show()

#Define the model architecture
input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)


#Define the loss function
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

#Do a pass and update the weights once
print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)


#Train the model with stochastic gradient descent
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

#Test the model
correct_count, all_count = 0, 0
for images,labels in valloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

#Generate a figure to show the test accuracy
# Test accuracy
test_accuracy = correct_count / all_count  # You already have this value

# Labels and values for the bar chart
labels = ['Model 1']
values = [test_accuracy]

# Generate bar chart
plt.figure(figsize=(5, 6))
bars = plt.bar(labels, values, color=['green'])

# Add value annotations on the bar
yval = bars[0].get_height()
plt.text(bars[0].get_x() + bars[0].get_width()/2.0, yval, round(yval, 2), va='bottom')  # va: vertical alignment


# Add axis labels and title
#plt.xlabel('Model 1')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

# Save as SVG
plt.savefig('Test_Accuracy1.svg', format='svg')

# Show the plot
plt.show()