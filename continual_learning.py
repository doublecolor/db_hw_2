import torch
from torchvision import datasets, transforms

from skimage.transform import resize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the CNN module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #design CNN model for SVHN dataset
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_dight(self, x):
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()
        
def convert_mnist_to_svhn(mnist_dataset):
    # Convert MNIST dataset to SVHN format
    svhn_dataset = []
    for image, label in mnist_dataset:
        image = image.squeeze().numpy()
        image = resize(image, (32, 32), mode='constant')
        # print(image.shape)
        image = np.stack((image,)*3, axis=-1)
        # print(image.shape)
        # image = (image - 0.1307) / 0.3081
        image = torch.tensor(image).permute(2, 0, 1)
        svhn_dataset.append((image, label))

    return svhn_dataset
        
def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    # train model with dataset
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished training')

    # Save the trained model
    torch.save(model, 'pretrained_model_torch.pth')

    import joblib
    joblib.dump(model, 'pretrained_model_joblib.pth')

    import pickle
    with open('pretrained_model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)

def ab_testing(dataloader_mnist, dataloader_svhn):
    # Evaluate the model on the MNIST
    model = torch.load('pretrained_model_torch.pth')
    model.eval()
    mnist_correct = 0
    with torch.no_grad():
        for images, labels in dataloader_mnist:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            mnist_correct += (predicted == labels).sum().item()
    mnist_accuracy = mnist_correct / len(dataloader_mnist.dataset)
    print(f'MNIST Accuracy: {mnist_accuracy}')

    # Evaluate the model on the SVHN
    svhn_correct = 0
    with torch.no_grad():
        for images, labels in dataloader_svhn:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            svhn_correct += (predicted == labels).sum().item()
    svhn_accuracy = svhn_correct / len(dataloader_svhn.dataset)
    print(f'SVHN Accuracy: {svhn_accuracy}')

    return mnist_accuracy, svhn_accuracy


def main():
    # download the MNIST and SVHN datasets from onilne  
    

    # Load the MNIST dataset
    mnist_train_dataset = datasets.MNIST('../data/mnist', train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    mnist_test_dataset = datasets.MNIST('../data/mnist', train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
    
    expanded_mnist_train_dataset = convert_mnist_to_svhn(mnist_train_dataset)
    expanded_mnist_test_dataset = convert_mnist_to_svhn(mnist_test_dataset)

    # Load the SVHN dataset
    svhn_train_dataset = datasets.SVHN('../data/svhn', split='train', download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    svhn_test_dataset = datasets.SVHN('../data/svhn', split='test', download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))

    # Create data loaders
    mnist_train_loader = torch.utils.data.DataLoader(expanded_mnist_train_dataset, batch_size=64, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(expanded_mnist_test_dataset, batch_size=64, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_train_dataset, batch_size=64, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_test_dataset, batch_size=64, shuffle=False)

    # Initialize the CNN module
    model = CNN()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Train the model on the MNIST dataset
    train_model(model, mnist_train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model on the MNIST test set
    print('Evaluating the model on the MNIST and SVHN test set')
    ab_testing(mnist_test_loader, svhn_test_loader)

    # Train the model on the SVHN dataset
    train_model(model, svhn_train_loader, criterion, optimizer, num_epochs=10)

    print('Evaluating the model on the MNIST and SVHN test set')
    ab_testing(mnist_test_loader, svhn_test_loader)

    # Save the trained model
    torch.save(model, 'pretrained_model_torch.pth')

    # Save the trained model using joblib
    import joblib
    joblib.dump(model, 'pretrained_model_joblib.pth')

    # Save the trained model using pickle
    import pickle
    with open('pretrained_model_pickle.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    print('Continual learning script executed as main program')
    main()    