from concrete_cnn import concrete_cnn
from utils import check_device

if __name__ == "__main__":
    num_epochs = 1000
    batch_size = 64
    device = check_device()

    concrete_cnn(device, num_epochs, batch_size)
