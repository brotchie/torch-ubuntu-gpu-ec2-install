# Installing Torch on Ubuntu 14.04 Amazon EC2 GPU Instances
This is a guide for installing the Torch machine learning ecosystem onto a GPU EC2 instance running Ubuntu 14.04.

**Note:** I have created and made available a Community EC2 AMI following these step with the name `torch-ubuntu-14.04-cuda-7.0-28` and ami-id `ami-c79b7eac`. Simply search for `ami-c79b7eac` in Community AMIs when creating an instance to get up and running quickly.

Preliminary steps:

 - Start a `g2.2xlarge` or `g2.8xlarge` instance with the *Ubuntu Server 14.04 LTS (HVM), SSD Volume Type - ami-d05e75b8* base AMI;
 - On Step 4: Add Storage of the instance configuration, increase storage on the primary volume from 8GB to 16GB; the 8GB default is too small;
 - Ensure the SSH port is allowed in the security group;
 - SSH into your running instance.


Note that the latest version of the CUDA .deb package is available at https://developer.nvidia.com/cuda-downloads.

```bash
# Pull the latest NVIDIA CUDA package and install it. Note that this step
# simply installs a local repository. It doesn't actually install the cuda
# toolkits / drivers.
wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb

# Update apt repositories and install the linux-image-extra-virtual package.
# This package include the drm.ko kernel module that's required by the NVIDIA drivers.
# When prompted during install, choose "install the package maintainer's version"
# to ensure the latest version of the Linux kernel is booted.
sudo apt-get update
sudo apt-get install -y linux-image-extra-virtual

# Install the version of the headers that matches the freshly installed kernel
# from the previous step.
sudo apt-get install -y linux-source linux-headers-3.13.0-53-generic

# Now we can install the cuda toolkits and drivers. The installation process
# will automatically compile the driver kernel modules.
sudo apt-get install -y cuda

# We now have to reboot to load the new kernel and kernel drivers.
sudo reboot

# Add the cuda binaries and shared library path to your .bashrc
cat >> ~/.bashrc <<END
export PATH=/usr/local/cuda-7.0/bin/:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64/:\$LD_LIBRARY_PATH
END
source ~/.bashrc

# Install and compiled the deviceQuery sample from the cuda distribution
# to validate the NVIDIA driver installation was successful.
cd ~
cuda-install-samples-7.0.sh .
cd NVIDIA_CUDA-7.0_Samples/1_Utilities/deviceQuery/
make
./deviceQuery

# Run the torche easy one-line install. This will install the cuda accelerated
# tensor and neural net lua packages.
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

# Install most of the dependencies for the ipython notebook.
sudo apt-get install -y ipython-notebook python-pip python-dev
# Remove the Ubuntu ipython and install the latest version from pip
sudo apt-get remove ipython ipython-notebook
sudo pip install ipython

# Install additional deps required by the latest ipython.
sudo pip install tornado --upgrade
sudo pip install jsonschema
sudo pip install terminado

# Finally install itorch.
sudo luarocks install itorch

# Fix up some permissions because we installed torch as root.
sudo chown -R ubuntu:ubuntu ~/.ipython
```

## Testing GPU Processing with a Recurrent Neural Network

Testing with a real-world example. Here we use Andrej Karpathy's code from [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) available at https://github.com/karpathy/char-rnn.

```bash
$ sudo luarocks install nngraph 
$ sudo luarocks install optim

$ git clone https://github.com/karpathy/char-rnn
$ cd char-rnn

# Train the RNN with the shakespeare dataset.
$ th train.lua

```

You should see Torch execute 6330 batches of training. On a `g2.2xlarge` instance it should take around 130ms per batch. After the model has trained, you can sample text from the RNN using:

```bash
$ th sample.lua cv/some_checkpoint.t7
```

