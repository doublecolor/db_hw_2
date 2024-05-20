import torch
from torchvision import datasets, transforms

from skimage.transform import resize

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb

wandb.login(key='905a03c6f175d2e773d5a505f4806f09cd514e03')
wandb.init(project='db_hw_3_youngmok_kim')

# Define the CNN module
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def predict_dight(self, x):
        with torch.no_grad():
            output = self(x)
            _, predicted = torch.max(output.data, 1)
            return predicted.item()
        
# Sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10, 15]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.1
        },
        'classes': {
            'values': [10]  # 예를 들어 MNIST 데이터셋을 사용하는 경우 10개의 클래스를 갖습니다.
        }
    }
}

def group_data_by_digit(dataset, grouped_data, grouped_labels):
    for image, label in dataset:
        if label in [0, 1]:
            key = "01"
        elif label in [2, 3]:
            key = "23"
        elif label in [4, 5]:
            key = "45"
        elif label in [6, 7]:
            key = "67"
        elif label in [8, 9]:
            key = "89"
        
        grouped_data[key].append(image)
        grouped_labels[key].append(label)

# download the MNIST dataset
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
    
# Create dictionaries to store the groups
grouped_data = {f"{i}{i+1}": [] for i in range(0, 10, 2)}
grouped_labels = {f"{i}{i+1}": [] for i in range(0, 10, 2)}

group_data_by_digit(mnist_train_dataset, grouped_data, grouped_labels)
group_data_by_digit(mnist_test_dataset, grouped_data, grouped_labels)

for key in grouped_data:
    grouped_data[key] = torch.stack(grouped_data[key])
    grouped_labels[key] = torch.tensor(grouped_labels[key])

class GroupedDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
split_dataloaders = {}
for key in grouped_data:
    dataset = GroupedDataset(grouped_data[key], grouped_labels[key])
    split_dataloaders[key] = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Create data loaders
mnist_train_loader = torch.utils.data.DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)

# Initialize the CNN module
model = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train_model():
    wandb.init()
    config = wandb.config

    model = CNN()
    trainloader = mnist_train_loader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    
    wandb.watch(model, criterion, log='all', log_freq=10)

    # train model with dataset
    for epoch in range(config.num_epochs):
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
                wandb.log({'epoch': epoch + 1,'loss': running_loss / 100})
                running_loss = 0.0

    print('Finished training')
    wandb.log({'final_loss': running_loss})

def train_model_continual():

    wandb.init()
    config = wandb.config

    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    split_dataloaders = split_dataloaders
    key = key
    
    wandb.watch(model, criterion, log='all', log_freq=10)

    for key in grouped_data:
        # train model with dataset
        for epoch in range(config.num_epochs):
            running_loss = 0.0
            for i, data in enumerate(split_dataloaders[key], 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    wandb.log({'epoch': epoch + 1,'loss': running_loss / 100})
                    running_loss = 0.0

    print('Finished training')
    wandb.log({'final_loss': running_loss})

def test(model, dataloader_mnist):
    # Evaluate the model on the MNIST
    
    model.eval()
    mnist_correct = 0
    with torch.no_grad():
        for images, labels in dataloader_mnist:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            mnist_correct += (predicted == labels).sum().item()
    mnist_accuracy = mnist_correct / len(dataloader_mnist.dataset)
    print(f'MNIST Accuracy: {mnist_accuracy}')

    wandb.log({'test_accuracy': mnist_accuracy})
    torch.onnx.export(model, images, 'model.onnx')
    wandb.save('model.onnx')

    return mnist_accuracy


def main():
    
    sweep_id = wandb.sweep(sweep_config, project='db_hw_3_youngmok_kim')
    # Train the model on the MNIST dataset
    wandb.agent(sweep_id, function=train_model, count=10)
    wandb.agent(sweep_id, function=train_model_continual, count=10)

    # Evaluate the model on the MNIST test set
    print('Evaluating the model on the MNIST and SVHN test set')
    test(mnist_test_loader)

    # Save the trained model
    torch.save(model, 'pretrained_model_torch_hw3.pth')

    # Save the trained model using joblib
    import joblib
    joblib.dump(model, 'pretrained_model_joblib_hw3.pth')

    # Save the trained model using pickle
    import pickle
    with open('pretrained_model_pickle_hw3.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    print('Continual learning script executed as main program')
    main()    