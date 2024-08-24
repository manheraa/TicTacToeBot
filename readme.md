
This `README.md` provides clear instructions on how to run the game, explains the model architecture, and gives an overview of the training process and performance.
# Tic-Tac-Toe GCN Model

This project provides a Graph Convolutional Network (GCN) model trained to predict the outcome of a Tic-Tac-Toe game based on the current board state. You can either play the game through the command-line interface (CLI) or via a Flask-based web interface.

## Installation

To get started, clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
python app.py

CLI Version
To play the game in the command-line interface, run:
python new_node.py```

## Model Archiecture

-Input Layer: The input dimension corresponds to the number of features per node, which is typically 9 in this Tic-Tac-Toe scenario.
-Hidden Layers: The GCN model has two convolutional layers (conv1 and conv2) that aggregate information from neighboring nodes. Each layer applies a non-linear activation function (ReLU).
-Output Layer: The fully connected layer (fc) maps the learned features to the output space, representing the three possible outcomes (win, loss, draw).


