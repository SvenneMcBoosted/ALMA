import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import timeit


n_epochs = 10

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(8192, 128)  # Are these optimal? 
        self.fc2 = nn.Linear(128, 2)  # Are these optimal?

#  TODO: Add more layers when dataset is larger
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 29 * 29, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 2)

#     def forward(self, x):
#         x = self.pool(nn.functional.relu(self.conv1(x)))
#         x = self.pool(nn.functional.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 29 * 29)
#         x = nn.functional.relu(self.fc1(x))
#         x = nn.functional.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(128 * 14 * 14, 512)
#         self.fc2 = nn.Linear(512, 2)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool1(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool2(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool3(x)
#         x = x.view(-1, 128 * 14 * 14)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Define the transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # nr.1 (torch default)
    # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]) # nr.2
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # nr.3
])

# Load the dataset
train_set = datasets.ImageFolder(root='../../data/png/train/', transform=transform)
num_classes = len(train_set.classes)
# print(num_classes)  # 2
train_loader = torch.utils.data.DataLoader(train_set, batch_size=2, shuffle=True)

# Create the model
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Maybe do weighted entropy loss function towards the smaller class in train data
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # lr=0.001, 0.01, 0.1

t_0 = timeit.default_timer()  # Timer start
print("Training started")
# Train the model
# n_epochs = 10
total_images = 0
print(f'Number of epochs: {n_epochs}')
print('*'*50)
for epoch in range(n_epochs): # Trying between 1-15 bc hardware bottleneck
    running_loss = 0.0
    epoch_images = 0  # count images trained in this epoch
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_images += len(inputs)  # increment counter by batch size
        total_images += len(inputs)  # increment total counter
        if i % 2000 == 1999:  
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Epoch %d trained on %d images' % (epoch + 1, epoch_images))

print('*'*50)
print('Finished Training')
t_1 = timeit.default_timer()
# calculate elapsed time and print
elapsed_time = round((t_1 - t_0), 3)
print(f"Elapsed time: {elapsed_time} s")

# Test the model
test_set = datasets.ImageFolder(root='../../data/png/test/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=10, shuffle=False)

# Init some useful variables
correct = 0
total = 0
predictions = []
true_labels = []
misclassified = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.numpy())
        true_labels.extend(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # check for misclassified images and add their fpaths to the misclassified list
        for i in range(len(predicted)):
            if predicted[i] != labels[i]:
                filename = test_set.samples[i][0]
                true_label = test_set.classes[labels[i]]
                predicted_label = test_set.classes[predicted[i]]
                misclassified.append((filename, true_label, predicted_label))

print('*'*50)
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
print('*'*50)

# print out the misclassified images and their true/predicted labels
print("Misclassified images:")
for item in misclassified:
    print('-'*35)
    print('Image:', item[0])
    print('True label:', item[1])
    print('Predicted label:', item[2])

print('*'*50)
print("Confusion Matrix (CLI):")
# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions)
print(conf_matrix)
class_names = train_set.classes
print(class_names)

# Create the confusion matrix object
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

# Generate the plot of the confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
cm_display.plot(ax=ax)
plt.show()
