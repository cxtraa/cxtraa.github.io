---
title: 'Spam classification model'
layout: post
parent: Projects
nav-order: 1
---

# Creating a spam filter model using machine learning
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

This guide will walk you through creating your own machine learning model in PyTorch to classify spam emails, from the data collection and cleaning phase, all the way to deploying the model as a web app online. The goal of this model is simple: to identify user emails as spam or not spam.

Check out the <a href="https://web-production-0105.up.railway.app/">final web app here!</a>

## Data cleaning and conversion to tensors

I used the dataset <a href="https://www.kaggle.com/datasets/venky73/spam-mails-dataset">here</a> on Kaggle. The dataset is a `.csv` file with two relevant columns: the text content of the emails, and their associated labels (spam or not spam).

We can read in the dataset using `pandas` as follows:

``` python
import pandas as pd

df = pd.read_csv("spam_dataset.csv")
```

We only need two columns: `text` and `label_num`:

``` python
df = df[['text', 'label_num']]
```

Also, let's remove all entries that have `NaN` values:

``` python
df = df.dropna()
```

Next, convert the dataset to a NumPy matrix:
``` python
import numpy as np

np_dataset = df.to_numpy()
```

Next, I want to create two Python `List` objects. The first will be a list of strings, the second will be a list of labels converted to integers:

``` python
texts = list(np_dataset[0, :])
labels = list(np_dataset[1, :].astype(int))
```

For this project, we are going to leverage the `sentence_transformers` library. This allows us to use a pre-trained **embedding model** on our email texts.

Embedding the texts means converting them to numerical vectors in a high-dimensional space. The model we are using today uses 768-length vectors. The embeddings are found by using an encoder-only transformer model such as BERT. First, the text input is tokenized, and each token is embedded into a 768-length vector in the transformer's residual stream. Through the self-attention mechanism, each of the 768-length vectors (corresponding to 1 token each) copies information from its neighbours, allowing them to store sentence context. Once at the end of the residual stream, the token vectors are all averaged to find a "representative vector" for the whole sequence. These models take a long time to train, and require a large amount of text to work well, which is why we are using the pre-trained `paraphrase-MiniLM-L12-v2` model from `sentence_transformers`:

``` python
from sentence_transformers import SentenceTransformers

embedding_model = SentenceTransformers("paraphrase-MiniLM-L12-v2")
embedded_texts = embedding_model.encode(texts) # Returns a NumPy array of shape (num_samples, embedding_dim)
```

This will take a minute or two to run, but is well worth it, because now we have vectors for each email that are all the same length, which makes it much easier to run them through a model. We're going to use these embedding vectors as the input for a neural network which will perform classification on the emails. The idea behind using an embedding model is that we hope that the spam vectors and non-spam vectors will be in different subspaces in the 768-dimensional vector space, which should make it easier for the neural network to classify them. 

Now, we are going to convert the embeddings and the labels to tensors:
``` python
import torch as t

embeddings_tensor = t.tensor(embedded_texts) 
labels_tensor = t.tensor(labels, dtype=t.long)
```

Let's create a `TensorDataset` instance from these tensors:
``` python
from torch.utils.data import TensorDataset

tensor_dataset = TensorDataset(embeddings_tensor, labels_tensor)
```

Also, we should split the dataset into training, validation, and test sets:
``` python
from torch.utils.data import random_split

# Adjust as you please
train_frac = 0.7
val_frac = 0.15

num_samples = len(tensor_dataset)
train_size = int(train_frac * num_samples)
val_size = int(val_frac * num_samples)
test_size = num_samples - train_size - val_size

train_set, val_set, test_set = random_split(tensor_dataset, [train_size, val_size, test_size])
```

Now we can create our `DataLoader` objects for the train set, val set, and test set. This will make it easy to take batches from the datasets during training:
``` python
from torch.utils.data import DataLoader

# Set to some reasonable power of 2
batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # We shuffle the training data so the model doesn't learn the order of the answers!
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

## Creating a neural network

Creating a neural network is very simple with PyTorch. We're going to define a fully-connected, feedforward neural network with `ReLU` activations. You can experiment with different activations such as `LeakyReLU`, or perhaps add `Dropout` layers to improve generalisation, but I found that these offered no discernible improvements in the model's performance.

``` python
from torch import nn

class Model(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.model(x)
```

The `Model` inherits from `nn.Module`, allowing it to benefit from PyTorch methods. It has 2 hidden layers, with 256 and 128 nodes respectively. The output layer has 2 nodes; these are the logits for the two output classes - 0 (not spam) and 1 (spam). The `forward` method must be implemented for any model class in PyTorch; it defines what the model should do with the input `x`. Since we used `nn.Sequential` to define all the model layers, this was a very simple task.

Let's instantiate the model:
``` python
model = Model(embedding_dim=embedding_model.get_sentence_embedding_dimension())
```

## Training the model

To train the model, we need a few ingredients:
1. An optimiser, such as `SGD`, `Adam`, or `AdamW`.
2. A loss function.
3. A training loop, ideally with helper functions for doing one training step or one validation step.

We will use the `Adam` optimiser. Roughly speaking, it has a different learning rate for each parameter. If the gradient of the loss with respect to some parameter `w_i` is changing sign a lot, it means the optimiser is oscillating in this direction, so we should reduce the learning rate. Conversely, if the gradient is not changing sign, we can slightly increase the learning rate. The `Adam` optimiser has been proven to have good convergence in the machine learning literature. We also use some weight decay (L2 regularisation) which applies a penalty for large weights (specifically, it adds the L2 norm of the weight vector to the loss function). This has been shown to reduce overfitting. Experiment with different weight decay values to see what works!

``` python 
weight_decay = 1e-03

optimiser = t.optim.Adam(model.parameters(), weight_decay=weight_decay)
```

For a classification task such as this, the cross entropy loss is a natural choice. PyTorch's implementation of the cross entropy loss automatically softmaxes the logits to produce a probability distribution, so we don't have to softmax ourselves. Furthermore, the labels need not be probability distributions, PyTorch will infer the labels as indices where the distribution has a 1, and 0 everywhere else.

``` python
criterion = nn.CrossEntropyLoss()
```

Let's define some boilerplate helper functions `train_one_epoch` and `eval`:

``` python
def train_one_epoch(model, optimizer, criterion, train_loader):
    train_loss = 0
    for inputs, labels in train_loader:
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward() # Backpropagate loss to find gradients
        optimizer.step() # Perform one step of gradient descent
        optimizer.zero_grad() # Zero all parameter gradients so they don't accumulate
        train_loss += loss.item()
    return train_loss / len(train_loader)

def eval(model, criterion, val_loader):
    val_loss = 0
    for inputs, labels in val_loader:
        logits = model(inputs)
        loss = criterion(logits, labels)
        val_loss += loss.item()
    return val_loss / len(val_loader)
```

Finally, let's write a simple training loop that will report on the train loss and validation loss at every epoch:

``` python
train_losses = []
val_losses = []
num_epochs = 10

for epoch in range(1, num_epochs+1):
    train_loss = train_one_epoch(model, optimizer, criterion, train_loader)
    val_loss = eval(model, criterion, val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch}/{num_epochs}, Train loss : {train_loss:.3e}, Val loss : {val_loss:.3e}")
```

I was able to reach a minimum validation loss of around `6.5e-02`.

We can visualise the evolution of the losses using `matplotlib.pyplot` (or you can use a nicer plotting library, such as `Plotly.js`):
``` python
import matplotlib.pyplot as plt

train_val_fig = plt.figure()
epochs = list(range(1, num_epochs+1))
plt.plot(epochs, train_losses)
plt.plot(epochs, val_losses)
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend(["Train", "Val"])
plt.title("Loss evolution during training of spam filter model")
plt.show()
```

## Inference and testing

We can test the model performance on the test set by writing another helper function `eval_test`. This will return the model accuracy (the fraction of correctly classified samples) rather than the cross entropy loss as a percentage:

``` python
def eval_test(model, test_set):
    num_correct = 0
    num_samples = len(test_set)
    for input, label in test_set:
        logits = model(input)
        prediction = t.argmax(logits, dim=-1) # Get the index of the class with highest probability according to the model
        if prediction == label:
            num_correct += 1
    print(f"Accuracy : {round(num_correct/num_samples, 3) * 100}%")
    return num_correct / num_samples
```

The model gets 100% accuracy on the test set, which is a sign it generalised to the dataset pretty well. However, this isn't necessarily indicative that the model will do good on completely new spam emails from a different dataset, because this dataset might have been biased / missing something in some way from the population of all emails.

Let's save the model weights so we don't lose them:

``` python
from datetime import datetime

time_created = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"model_{time_created}.pth"
t.save(model.state_dict(), f"./{filename}")
```

## Creating a utils.py file

So far, we've just been working in a Jupyter Notebook. If we want to turn this into a web app, we will need to have a `utils.py` file which contains the `Model` class template and a `predict` function which takes in a string and returns the model's prediction. Here's what my `utils.py` file looks like:

``` python
import torch as t
import numpy as np
from torch import nn

class Model(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    def forward(self, x):
        return self.model(x)
    
def predict(model, embedding_model, text):
    embedding = t.tensor(embedding_model.encode(text))
    logits = model(embedding)
    return t.argmax(logits).item()
```

When we develop our web app, we'll be able to import `Model` and `predict` from `utils`.

## Creating the web app using Flask

`Flask` is a lightweight framework that allows you to manage a web app's backend using Python. First, we'll setup a simple webpage using HTML/CSS and Bootstrap. I've provided my HTML here - feel free to re-style it as you wish. It consists of a title, a text box to enter the email contents, and a submit button. Name the file `index.html`, and place it inside a folder called `templates` located in the root of your directory.

``` html
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Classifier</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="content">
            <div class="text-center container w-50">
                <h1 class="text-center mb-4">SPAM CLASSIFIER</h1>
                <form class="mb-4" method="POST">
                    <div class="mb-3">
                        <label for="input-email" class="form-label">Check if your email is spam!</label>
                        <textarea name="input-email" id="input-email" class="form-control" placeholder="Type here..." rows="10"></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary w-75">Submit</button>
                </form>
        </div>
        <footer class="container">
            <p>© 2024 Moosa Saghir</p>
            <p>Read about this model on my <a href="https://cxtraa.github.io/">website.</a></p>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    </body>
</html>
```

Next, let's build the backend using Flask. To get started, create a file called `app.py` in the root directory:

``` python
from flask import Flask, request, render_template
from utils import Model, predict
from sentence_transformers import SentenceTransformer
import torch as t

app = Flask(__name__)

# Load the embedding model, network, and load the saved weights into the network
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
model = Model(embedding_dim=embedding_model.get_sentence_embedding_dimension())
model.load_state_dict(t.load("./model.pth", map_location="cpu"))

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
```

For this project, Flask acts as the web server. Therefore, `GET` requests request data from the Flask app, and `POST` requests send data to the Flask app, which we can manage in `app.py`.

Let's break this code down:
- `Flask` is a class. When instantiated, it becomes the web server for our web app.
- `@app.route()` is a decorator that associates the function `index()` with it. Whenever a `GET` or `POST` request is made at the root (index) page, `index()` is called. 
- `render_template("index.html")` tells the web server to load `index.html`.
- `app.run(debug=True)` runs the webpage in debug mode. A local link will be printed to the console where you can view your webpage.

When the user submits their email on the web page, a `POST` request is made to the server. We can access the `input-email` field form the request, and run it through our model using `predict`, which we wrote in `utils.py` earlier. Finally, we call `render_template`, but we pass in an additional argument - `prediction`.

``` python
@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    if request.method == "POST":
        email_text = request.form["input-email"]
        prediction = predict(
            model=model,
            embedding_model=embedding_model,
            text=email_text,
        )

    # Load index.html, and pass the prediction to the page
    return render_template("index.html", prediction=prediction)
```

The additional argument `prediction` can be used to dynamically create a results box which either says "Spam" or "No spam" depending on whether `prediction` is 1 or 0. Let's add that in our HTML file (you can decide where exactly in the page you want the results box to appear):

{% raw %}
``` html
{% if prediction is not none %}
    <div class="container w-75 alert {% if prediction == 0 %}alert-success{% else %}alert-danger{% endif %}">
        {% if prediction == 0 %}
        We've run your email through our model, and we don't think this is spam.
        {% else %}
        Our model predicts that this email is probably spam.
        {% endif %}
    </div>
{% endif %}
```
{% endraw %}

And with that, the web app is functionally complete!

## Deployment

You can deploy a simple web app like this, with low traffic at <a href="https://railway.app/">Railway</a>, which offers free hosting upto $5 in compute costs per month. After your monthly $5 quota runs out, the server automatically stops; you don't get charged.

If you want to do this, you'll want to create a GitHub repo for your app, making sure not to include the initial training notebook. Also create a `.gitignore` file and add the following files:
```
__pycache__/
spam.csv
train_notebook.ipynb
```

Next, download the `gunicorn` library:

```
pip install gunicorn
```

And create a file called `Procfile` in the root (with no file extension), and add the following:

```
web: gunicorn app:app
```

You'll also want to create a `requirements.txt` file which lists all the Python libraries that need to be installed for your project to run. You can do this by running the following command in the Terminal:

```
pip freeze > requirements.txt
```

Railway will connect to your GitHub account and automatically build the project for you, and host it live for other people to use!

For reference, <a href="https://web-production-0105.up.railway.app/"> here is the web app I created.</a>

## Conclusion

In this guide, I explained how you can create a whole machine learning project, right from data cleaning, processing, down to deploying the model itself as a web app. More projects will be coming soon, so look out for those!