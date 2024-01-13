---
title: 'Tutorial: CNN model with hyperparameter sweep'
layout: post
parent: Artifical intelligence
nav-order: 3
---

# Training a CNN model and running a hyperparameter sweep with wandb
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

This page demonstrates how to easily setup two different models to train an image classifier on the standard MNIST dataset. The first is a dense neural network with ReLU activations, and the second is a CNN (convolutional neural network).

The CNN is the superior architecture because it involves far fewer weights allowing greater generalisability and takes into account the fact that pixels close together are related. The traditional linear network is provided for reference.

Finally, we use the `wandb` library (Weights and Biases) which allows you to plot metrics such as validation loss and accuracy in real-time and produce clean plots online. More importantly, `wandb` has support for *hyperparameter search* - that is, we can trial a range of hyperparameters such as learning rate and batch size, and identify the optimal values for our model.

To use `wandb`, sign up online [here](https://wandb.ai/login).

PyTorch is used here as the main machine learning framework. The dataset is sourced from the `torchvision` library which also provides some useful image transformations such as `.toTensor()`. We also import `tqdm` so we can produce a dynamic progress bar during training.

# Data loading and preprocessing

Firstly, we install all libraries and dependencies required. We then tell PyTorch to use `cuda` (GPU) if it is available, and otherwise to use the CPU (which is much slower).

Next, we use the `torchvision` library to define our image transforms. These consist of two operations: firstly, we convert the images to tensors with their pixel values scaled to $$ [0,1] $$, and secondly, we normalize the values with mean 0.5 and standard deviation 0.5. 

Using `torchvision` datasets, we import the classic MNIST dataset. Then, we define our train-validation split, which we've set to be 90:10 here. Finally, we create PyTorch `DataLoaders`. These are extremely useful objects which feed in data to our model in batches of size `batch_size`, and allow optional shuffling of data. We shuffle the data in training to prevent the model from overfitting, but shuffling isn't necessary for validation because the model doesn't train from the validation set.

```python
pip install einops
pip install wandb
pip install tqdm
```

```python
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import einops
import wandb
from tqdm import tqdm
```

```python
device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

```python
transform = transforms.Compose([
    transforms.ToTensor(),  # convert image to tensor with pixels in [0,1]
    transforms.Normalize((0.5,), (0.5,)) # mu = 0.5, sigma = 0.5
])

train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_size = int(0.1 * len(train_set))
train_size = len(train_set) - val_size
train_subset, val_subset = random_split(train_set, [train_size, val_size])

test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = t.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = t.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = t.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
```

# A bit about PyTorch modules and CNNs

## Setting up a PyTorch module

PyTorch allows the user to create their own `nn.Module` classes. For example, we create a `CNN()` class below. In the constructor (`__init__`), we must use the `super()` method to inherit the functionality of the `nn.Module` class. This will allow our class to work seamlessly with the PyTorch framework.

We then use a `nn.Sequential` object to store all of our model layers. In this way, we can highly efficiently produce our model architecture in just a few lines of code!

Finally, PyTorch classes must include a `forward()` method, which takes a tensor `x` as input. Here, we simply apply `self.model()` to `x`, and return the result!

Now, to create our model later in the code, we can just do:

``` python
my_model = CNN()
```
Super easy!

## Convolutional Neural Networks (CNN)

Convolutional neural networks apply a convolution operation to each of the image channels. In our case we have grayscale images, so there is only one `in_channel`. For intuition on the convolution operation, see this [excellent video by 3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA&pp=ygUYM2JsdWUxYnJvd24gY29udm9sdXRpb25z).

The `out_channels` represents the number of channels we will have after applying the operation. In our case, we go from 1 -> 32 channels, so that means we apply 32 kernels (matrices), one by one over the image, producing 32 images.

A kernel is simply the matrix we apply to convolve over the image. We use `3*3` here, which is standard for a CNN.

`stride` represents the amount by which the kernel moves each step. So if the stride is 1, it means we shift the kernel by one pixel each iteration of the convolution. Finally, `padding` is used to apply a border of pixels to the outside of the image. This is because otherwise, we would lose pixels on the edge of the image each convolution.

**Note: both a standard and convolutional model are shown below. Remember to only use one in your actual code!**


```python
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(

        # CONVOLUTIONAL MODEL
        # initial shape: (c, h, w) = (1, 28, 28)

        # Conv layer 1
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # shape now (32, 14, 14) due to MaxPool layer

        # Conv layer 2
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        # shape now (64, 7, 7)

        nn.Flatten(),

        # shape now (64*7*7,)

        # Dense standard NN layer
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, 10)  # for the 10 digit classes

        # DENSE MODEL
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 10)
    )

  def forward(self, x):
    return self.model(x)
```

# Setting up a training loop

Now that we have set-up our `CNN` class, we need a function that processes data in batches from our DataLoader, performs gradient descent (using the Adam optimiser), and calculates the loss. Thankfully, this can all be done with relatively little effort on our part, thanks to PyTorch.

`train_loader` is our DataLoader for the training data. `optimizer` is our gradient descent optimizer, which in this case is `torch.optim.Adam`. Finally, we keep track of the epoch number as well.

We then create a nice progress bar using `tqdm`, which shows progress within an epoch. Make sure to set `position=0` otherwise a new bar will be generated for every batch, which gets very cluttered quickly!

Next, we iterate through the data in `train_loader`. `inputs` represents the image data as a `t.Tensor()`, and `labels` stores the true classifications. Notice that we immediately send both of these to `device`, which should be `cuda` (GPU) by default to improve runtime.

We then perform the following operations:
- Set gradients to zero (the gradients from the previous SGD step are not automatically cleared, so we have to clear them to avoid accumulation)
- Calculate the predictions from our model
- Calculate the loss based on some loss function `loss_func` - we use cross entropy in this tutorial.
- Use backpropagation to calculate the gradients $ \frac{dL}{dw} $ for all weights $ w $.
- Update the model weights according to the Adam algorithm (pseudocode can be found on PyTorch documentation [here.](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)).

The `evaluate()` function is quite similar. Obviously, we don't include an optimiser because we are not trying to update anything during testing. To tell PyTorch that we are in testing mode, we use `with torch.inference_mode()`. We additionally calculate the number of correctly classified images and return the accuracy of the model.


```python
def train_one_epoch(train_loader, model, loss_func, optimizer, epoch):
  model.train() # set model to training mode
  train_loss = 0.0

  progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{hyperparams['num_epochs']}", position=0, leave=True)

  for inputs, labels in train_loader:
    inputs, labels = inputs.to(device), labels.to(device)

    # zero gradients to avoid accumulation
    optimizer.zero_grad()

    # forward pass
    outputs = model(inputs)
    loss = loss_func(outputs, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    progress.update(1)

  wandb.log({"train_loss" : train_loss})

def evaluate(validation_loader, model, loss_func):
  model.eval()  # set model to evaluation mode
  val_loss = 0.0
  correct = 0
  total = 0

  with t.inference_mode():
    for inputs, labels in validation_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = loss_func(outputs, labels)
      val_loss += loss

      pred = t.argmax(outputs, dim=1)
      correct += (pred == labels).to(int).sum()
      total += len(inputs)

  val_accuracy = 100 * (correct / total)
  return val_loss, val_accuracy
```

# Using wandb easily

## Basic setup

`wandb` will now allow us to plot our validation loss over epochs really easily!

First, assuming you made a `wandb` account earlier, typing `wandb.login()` will prompt you to go to a link to find an API key which you will have to enter to authorise yourself. We then initialise our `wandb` run using `wandb.init()`. We then make a dictionary for our hyperparameters `hyperparams` for easy access.

Next, we create our model by producing an instance of the `CNN` class, making sure to store this on the GPU. We also specify our loss function, which is the *cross entropy* function. The cross entropy of two probability distributions $$ P(x) $$ and $$ Q(x) $$ is: \\[ L_{CE} = -\sum_{i=1}^{n} P(x) \log(Q(x)) \\]

where $ P(x) $ is the true distribution and $$ Q(x) $$ is our model's distribution. Because the distribution $ P(x) $ is always all 0s except 1 for the correct class, this formula can be re-written as: \\[ L_{CE} = -\sum_{i=1}^{n} \log(Q(x)) \\]

The cross entropy differs to the [KL divergence](https://www.statlect.com/fundamentals-of-probability/Kullback-Leibler-divergence) (another related loss function) by a constant. The KL divergence is the *expected difference of the log of the probability distributions*, which I find to be more intuitive for understanding why we want to use cross entropy in the first place!

We specify our optimiser, which is Adam. Look out on [my website](https://cxtraa.github.io/) for an article on gradient descent algorithms, because I'll be going into much more depth with how Adam works there!

Finally, we create a loop that calls `train_one_epoch` `num_epochs` times, calculates the validation loss, and logs it using `wandb.log`. Finally, remember to finish a run with `wandb.finish()`.

```python
wandb.login()
wandb.init(project='MNIST CNN', entity='moosasaghir10')

hyperparams = {
    'learning_rate' : 0.001,
    'batch_size' : 64,
    'num_epochs' : 10,
}
```

```python
model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
```

```python
for epoch in range(hyperparams['num_epochs']):
  train_one_epoch(train_loader, model, loss_func, optimizer, epoch)
  val_loss, val_accuracy = evaluate(test_loader, model, loss_func)
  print(f"Val accuracy: {val_loss:.2f}")
  wandb.log({"val_loss" : val_loss})

wandb.finish()
```

## Visualising results

Running the code above will prompt `wandb` to start logging data. Once the run is finished, you will get a link to your results page. Below are the plots I produced for the training loss and validation loss.

![mnist-single-run](https://i.ibb.co/R0mtBwZ/Screenshot-from-2024-01-13-14-31-53.png)

Both the train loss and validation loss drop steadily throughout the run, meaning our model has no overfitting issues. Our model is a success!

# Running a hyperparameter search

We want to find the variables `num_epochs`, `batch_size`, that will minimise our validation loss. To do this, we can conduct a hyperparameter sweep using `wandb`. There are three options, `random`, `grid`, and `bayes`.

We will use Bayesian inference to find the optimal set of parameters. To do this, we need a prior, a truth, and a posterior. The prior reflects our current beliefs about what $$p(y \| x)$$ is, which says, "what is the probability that $$ y $$ is the correct set of parameters, given my evidence $$ x $$ - the validation loss?" In our case we are minimising `val_loss`, so we want to minimise $$ y $$. We can then write: \\[ p(y \| x) = \frac{p(x \| y) p(y)}{p(x)} \\]

Here, $$ p(y) $$ is our prior. It is our initial estimate of how likely this set of hyperparameters is. After each epoch, the model uses the evidence provided by the data $$ p(x) $$ which allows us to move in the direction of better hyperparameters. I will be writing a more detailed post on Bayesian statistics soon, so look out for that!

```python
sweep_config = {
    "method" : "bayes",
    "name" : "sweep",
    "metric" : {
        "name" : "val_loss",
        "goal" : "minimize",
    },
    "parameters" : {
        "learning_rate" : {
            "distribution" : "log_uniform_values",
            "min" : 1e-4,
            "max" : 1e-1
        },
        "batch_size" : {
            "values" : [32, 64, 128, 256],
        }
    }
}
```


```python
def train_wandb():

  with wandb.init() as run:
    config = run.config

    model = CNN()
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(model.parameters(), lr=config.learning_rate)

    # need to respecify our train loader and val loader because we are now changing the batch size

    train_loader = t.utils.data.DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = t.utils.data.DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    for epoch in range(hyperparams['num_epochs']):
      train_one_epoch(train_loader, model, loss_func, optimizer, epoch)
      val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
      print(f"Val accuracy: {val_loss:.2f}")
      wandb.log({"epoch" : epoch, "val_loss" : val_loss})

```


```python
sweep = wandb.sweep(sweep_config, project="MNIST CNN")
wandb.agent(sweep, train_wandb)
```

# Analysing the results

`wandb` produces a useful diagram that allows you to compare different hyperparameter combinations at a glance, like the one I've generated below for this tutorial:

![wandb-mnist-diagram](https://i.ibb.co/dp4gxFy/mnist-sweep-wandb.png)

Each line corresponds to a different combination of hyperparameters. As you can see, most of the combinations lead to near zero validation loss, however, one line stands out in particular. When the learning rate was very high (around 0.1), we had a massive validation loss.

We can conclude that a large learning rate is therefore not good for our model, whilst batch size seems to have basically no effect on the validation loss.

And that concludes the tutorial! Now you know how to train a simple CNN in PyTorch, log your results, and perform hyperparameter sweeps using `wandb`.
