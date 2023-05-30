import struct
import gzip
import numpy as np

import sys

sys.path.append('python/')
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    import struct, gzip
    import numpy as np
    with gzip.open("./data/train-labels-idx1-ubyte.gz", "rb") as f:
        magic, num = struct.unpack('<2I', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # print(labels)
    with gzip.open("./data/train-images-idx3-ubyte.gz", "rb") as f:
        magic, img_num, row_num, col_num = struct.unpack('<4I', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(-1, 28 * 28).astype("float32") / 255.0
        # print(images)
    ### END YOUR CODE
    return images, labels


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    batch_size = Z.shape[0]
    tmp1 = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    tmp2 = ndl.summation(ndl.multiply(Z, y_one_hot), axes=(1,))
    return ndl.summation(tmp1 - tmp2) / batch_size
    # return np.average(np.log(np.exp(Z).sum(axis=1)) - Z[np.arange(y.shape[0]), y])


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    m, n = X.shape[0], X.shape[1]
    batch_num = int(np.ceil(m / batch))  # 有多少个batch
    print(m, n, batch_num, batch)
    from time import time
    for i in range(1, batch_num + 1):
        if i == 100:
            aaa = 0
        start = time()
        if i != batch_num:
            s_index = (i - 1) * batch
            e_index = i * batch
        else:
            s_index = (i - 1) * batch
            e_index = m
        print("Batch Start:", s_index)
        X_batch = ndl.Tensor(X[s_index:e_index])
        y_batch = ndl.Tensor(y[s_index:e_index])
        z1 = ndl.relu(ndl.matmul(X_batch, W1))
        z2 = ndl.matmul(z1, W2)
        loss = softmax_loss(z2, y_batch)
        end1 = time() - start
        loss.backward()

        W1 = W1 - W1.grad * lr
        W2 = W2 - W2.grad * lr
        W1 = ndl.Tensor(W1.numpy())
        W2 = ndl.Tensor(W2.numpy())
        end2 = time() - start
        print("时间占比", 1 - end1 / end2)
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
