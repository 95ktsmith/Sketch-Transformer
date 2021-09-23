# Sketch-Transformer

By [Michael George](https://mag389.github.io/) and [Kevin Smith](https://github.com/95ktsmith/)

Sketch-Transformer is a project inspired by Google's [Sketch-RNN](https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn) and [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset. We wanted to utilize Transformer architecture in creating a model that can complete an unfinished drawing, akin to Sketch-RNN, like so:
![From circle to finished cat.](https://i.imgur.com/u2SG5D9.png)

## Why a Transformer?

Transformers are well known for their capabilities in the field of NLP, but they also work well for a multivariate time-series dataset, which is how these drawings are represented. As per Stroke-5 format, each drawing is a series of points where each point is a five feature vector of [x, y, p1, p2, p3] where x and y are offsets from the previous point, and [p1, p2, p3] is a one-hot vector for if the pen is down, up, or finished respectively.

## Model Architecture

Since our goal is to complete drawings, in essence just generate more points from a starting sequence, we're using only the Decoder structure of the Transformer. And given the two-objective nature of predicting both the offsets and the pen state for each point, we created a branched decoder model, as depicted here:

![Branched-Decoder Model](https://i.imgur.com/NP8eCqg.png)
We first use a linear layer to project the feature dimension to the model's dimension before adding positional encoding. From there, the encoded inputs are given to the base decoder blocks. The output of the base blocks is given to both the offset branch and the pen state branch to make their predictions. The model then returns (offset predictions, pen state predictions) as output.

The rationale behind branching the model in this way is to train the two branches independently for their respective tasks while being able to learn features shared by both. For offset predictions we use Mean Squared Error loss and for pen state predictions we use Categorical Cross-entropy. Gradient descent is performed on each branch using gradients calculated from their respective losses, while all gradients are applied to the base decoder blocks. 

At the moment, the model has only been trained on cats, but we aim to try other categories as well after fine-tuning results.

## Run it Yourself

A Google Colab notebook with executable examples can be found in the demo folder. The two weight files and cat.npz file will need to be uploaded to the notebook. 
