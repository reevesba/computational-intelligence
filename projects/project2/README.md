# Multi Layer Perceptron

### Objective 
Investigate the use of the Multi-Layer Perceptron (MLP, Sections 4.2.1 and 4.4.1 in textbook) to approximate the following function:

    f(𝑥, 𝑦)= sin⁡(π*10*𝑥 + 10/(1 + 𝑦²)) + ln⁡(𝑥² + 𝑦²), 
    where 1 ≤ 𝑥 ≤ 100 and 1 ≤ 𝑦 ≤ 100 are real numbers.

You may use any implementation of the MLP.

### Requirements
1.	Set up two sets of data, one for training and one for testing.
2.	Use the training set to compute the weights of the network. 
3.	Evaluate the approximation accuracy (the root-mean squared error) of the network on the test data. 
4.	Use a variable number of hidden layers (1-2), each with a variable number of neurons (0-19). 
5.	Investigate how the network’s performance is affected by varying the number of hidden layers and the number of hidden neurons in each layer. It is not necessary to try all 20 x 20 combinations of neurons in the two hidden layers; try only some of them.
6.	Investigate how the network’s performance is affected by varying the number of training patterns.

Hint: If you use your own implementation:
-	Use two input neurons: one for x and one for y.
-	For the hidden neurons, use bipolar or unipolar sigmoid activation functions.
-	Use a single output neuron, with the identity (linear) activation function (read Sections 3.2.3 and 4.4.1 from textbook).
-	Do not forget to add a fixed input (-1) to each layer


### Deliverables
-	Describe which implementation you used and, if it is your code, attach it
-	The approximation accuracy for the different MLP architectures used
-	Your conclusions about how the number of hidden layers and neurons influences the approximation error

### Directory Structure
- doc: Any relevant documentation/report.
- img: Hosts any images for this README.
- out: Program output. 
- src: Program source code files.