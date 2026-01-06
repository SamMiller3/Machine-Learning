# Artificial Intelligdnce

## Machine-Learning


### Reinforcement Learning

I have coded several Reinforcement Learning environments and algorithms to solve them such as TicTacToe and gridworlds with Q-Learning or Connect 4 with a DQN.

### Supervised Learning

#### From Scratch

I have implemented fundemental supervised learning algorithms from scratch

##### Multilayer Perceptron from scratch
I have implemented a multilayer perceptron from scratch using numpy, learning in more depth how backpropogation and gradient descent actually work as well as different activation and loss functions. It is implemented using OOP so different networks can be implemented if needed, currently one with an input layer with 2 nodes, a hidden layer with 3 nodes and an output layer with 1 node is implemented to simulate an XOR gate. If you give it training data for example

```
0,0,0
0,1,1
1,0,1
1,1,0
```

The first 2 columns being inputs and last column denoting an output it can be trained to simulate an XOR gate. Use at least 10,000+ epochs.

##### Others: 

I have also implemented algorithms such as linear regression, polynomial regression and support vector machine

#### PyTorch

##### Movie Review Sentiment Analysis:

I coded an MLP with 2 hidden layers to take a movie review and then output if it was positive or negative. It was trained on 50,000 movie reviews from IMDB.

##### MNIST Handwritten Digit Classifier

I coded an MLP to train on the MNIST dataset of handwritten digits to classify what digit they corresponded to.

## Symbolic AI

I have coded several Symbolic AI programs, including:

- TicTacToe MiniMax & Alpha-Beta Pruning
- Sudoku Solver using constraint propagation and heuristics 
- 1972 TicTacToe program by Allen Newell and Herbert Simon
