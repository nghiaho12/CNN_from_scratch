A convolutional neural network (CNN) written in C++ from scratch.
There is zero 3rd party library depenedicies.

I wrote it to jog my memory of how a CNN works.
Emphasis is placed on code readability and none on performance.
It is intended purely for educational purposes and not for serious use.

This code has been tested on Ubuntu 22.04.

# How to run
Only MNIST data format is supported. 

Download the MNIST handwritten dataset.

```
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
```

or alternatively the fashion MNIST dataset.

```
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

Decompress the files
```
gzip -d *.gz
```

Compile the code
```
mkdir build
cd build
cmake ..
make
./unit_tests
```

Train on the dataset
```
./train train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

```
Change the path to the files as needed.

