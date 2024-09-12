import torch
# from torch_kmeans import KMeans, ConstrainedKMeans
import os
import random
import matplotlib.pyplot as plt

import numpy as np
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor, Normalize

batch_size = 50
transform = Compose([ToTensor(),
    Normalize((0.5,), (0.5,))
    ])
dataset = FashionMNIST('MNIST_data/', train = True, transform = transform)
# print(dataset[0])

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.net =nn.Sequential( nn.Linear(768,768),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(768,768),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(768,768))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(1536, 768),
            nn.ReLU(inplace=True),
            nn.Linear(768, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.net.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.net(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # print('hello')
        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output


class APP_MATCHER(Dataset):
    def __init__(self, train=False):
        super(APP_MATCHER, self).__init__()

        self.dataset = FashionMNIST('MNIST_data/', download=True, train = train) # this part is unnessesary, im just cannibalizing the MNIST Dataset container

        self.load_training_data()

        self.data = self.dataset.data.flatten(start_dim=1).unsqueeze(1).clone()
        print(self.data[0].size())

        self.group_examples()

    def load_training_data(self):
        with open('final_detections.json') as fr:
            final_detections = json.load(fr)
        with open('labels.json') as fr:
            labels = json.load(fr)

        # final_detections.sort(key= lambda x: x['detection_id'])
        data = []
        targets = []
        for i in len(final_detections):
            person = -1
            for k in len(labels):
                if final_detections[i]['detection_id'] in labels[k]:
                    person = k
                    break
            if person == -1:
                break
            data.append(final_detections[i]['feature'])
            targets.append(person)
        
        self.dataset.data = torch.tensor(data)
        self.dataset.targets = torch.tensor(targets)

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group 
            examples based on class. 
            
            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in 
            `grouped_examples`, every value will conform to all of the indices for the MNIST 
            dataset examples that correspond to that key.
        """

        # get the targets from MNIST dataset
        np_arr = np.array(self.dataset.targets.clone())
        
        # group examples based on class
        self.grouped_examples = {}
        for i in range(0,10):
            self.grouped_examples[i] = np.where((np_arr==i))[0]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        """
            For every example, we will select two images. There are two cases, 
            positive and negative examples. For positive examples, we will have two 
            images from the same class. For negative examples, we will have two images 
            from different classes.

            Given an index, if the index is even, we will pick the second image from the same class, 
            but it won't be the same image we chose for the first class. This is used to ensure the positive
            example isn't trivial as the network would easily distinguish the similarity between same images. However,
            if the network were given two different images from the same class, the network will need to learn 
            the similarity between two different images representing the same class. If the index is odd, we will 
            pick the second image from a different class than the first image.
        """

        # pick some random class for the first image
        selected_class = random.randint(0, 9)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
        
        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        image_1 = self.data[index_1].clone().float()

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0]-1)
            
            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)
        
        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 9)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)

            
            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0]-1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            image_2 = self.data[index_2].clone().float()

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return image_1, image_2, target



def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)




def train(model, device, train_loader, n_epochs, lr=.0025):
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    criterion = nn.BCELoss()
    train_accuracy = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        train_correct = 0
        train_samples = 0
        for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images_1, images_2).squeeze()
            # print(targets.size())
            for i in range(len(targets)):
                train_samples += 1
                if  (targets[i] > 0.5 and outputs[i] > 0.5) or (targets[i] <= 0.5 and outputs[i] <= 0.5):
                    train_correct += 1
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_accuracy[epoch] = float(train_correct)/float(train_samples)
    
    return train_accuracy

def evaluate(input):
    model = SiameseNetwork()
    model.load_state_dict(torch.load('siamese_network.pt', weights_only=True))
    model.eval()
    return model.forward_once(input).squeeze()

def plot_accuracy(train_accuracies):
    """Plot accuracies"""
    plt.plot(train_accuracies, "-x")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Training"])
    plt.title("Accuracy vs. No. of epochs")
    plt.show()


def main():
    TRAIN = DataLoader(APP_MATCHER(train=True), batch_size=50)
    model = SiameseNetwork()
    device = get_default_device()
    to_device(model, device)
    history = train(model, device, TRAIN, 8, lr=.0025)
    torch.save(model.state_dict(), "siamese_network.pt")
    plot_accuracy(history)

# main()
