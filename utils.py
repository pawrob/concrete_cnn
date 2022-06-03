import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import transforms
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Dataset


def train_in_epochs(num_epochs, train_loader, device, optimizer, criterion, model):
    start = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):

            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)

            loss = criterion(scores, targets)
            optimizer.zero_grad()

            running_loss += loss.item()
            if batch_idx % 100 == 0:
                print('Epoch: {} Batch: {} loss: {}'.format(epoch, batch_idx, loss.item()))

            loss.backward()
            optimizer.step()

    print('Training Completed in: {} secs'.format(time.time() - start))



def check_acc_cnn(loader, model, device):
    num_correct = 0
    num_samples = 0
    pred = []
    targ = []

    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            scores = model(data)
            _, predictions = scores.max(1)

            # draw incorrect predictions
            # list_pred = predictions.tolist()
            # list_target = target.tolist()
            # for x in range(len(list_pred)):
            #     if not list_pred[x] == list_target[x]:
            #         imshow(torchvision.utils.make_grid(data[x]),f'AS IS: {list_pred[x]}, TO BE: {list_target[x]}')

            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)
            pred.extend(predictions.tolist())
            targ.extend(target.tolist())

        print(
            f'With training data {loader.dataset.train}, got {num_correct}/{num_samples} with acc: {float(num_correct) / float(num_samples) * 100:.2f}')
    model.train()
    cm = metrics.confusion_matrix(targ, pred)
    plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], normalize=False)



# check if cuda is available
def check_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


# plot img
def imshow(img, title):
    plt.title(title)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()


# plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Greens):
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title = title
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(i, j, format(cm[i, j], fmt), horizontalalignment="center",
                 color='white' if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel("true label")
    plt.xlabel("predicted label")
    plt.show()


def load_dataset(batch_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    dataset_train = Dataset(transform=transform, train=True)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size)
    dataset_test = Dataset(transform=transform, train=False)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size)

    return [train_loader, test_loader]
