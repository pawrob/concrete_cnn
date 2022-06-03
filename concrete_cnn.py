import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN
from pytorch_model_summary import summary
# from torchsummary import summary # for colab

from utils import train_in_epochs, check_acc_cnn, load_dataset


def concrete_cnn(device, num_epochs, batch_size):
    print('Training with CNN - Crack detection ')

    dataset = load_dataset(batch_size)
    model = CNN()
    model = model.to(device)
    print(summary(CNN(), torch.zeros(1, 3, 227, 227), show_input=True))
    # summary(model, (3, 227, 227)) # for colab

    criterion = nn.CrossEntropyLoss()  # loss function with softmax
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adaptive Movement Estimation algorithm

    train_in_epochs(num_epochs, dataset[0], device, optimizer, criterion, model)

    check_acc_cnn(dataset[0], model, device)
    check_acc_cnn(dataset[1], model, device)
