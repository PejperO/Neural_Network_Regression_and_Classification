# Neural Network Regression and Classification - A Comparative Study

This repository contains a collection of Python scripts that implement neural network models for both regression and classification tasks using custom-built neural networks and Keras-based networks. The goal of this project is to explore the use of neural networks for different types of data and tasks, examining the performance through key metrics and visualizations.

## Overview
### Files Included:
1. **4.a.fa.py:** A custom-built neural network implemented from scratch for regression using Sigmoid activation functions.
2. **4.a.py:** A similar custom-built network but using ReLU as the activation function.
3. **4.b.I.py:** A regression neural network built using the Keras Functional API, with early stopping and two hidden layers.
4. **4.b.II.py:** A classification neural network built using Keras to solve the classic moons dataset problem.

## Goals of the Project:
- Compare different activation functions (Sigmoid vs. ReLU) in a custom neural network for regression.
- Utilize Keras API to implement a more optimized neural network with early stopping for regression tasks.
- Train a neural network for a binary classification task using Keras and visualize decision boundaries.
- Understand the training behavior through visualizations like MSE, R², and accuracy during training.

## Running the Scripts
- **4.a.fa.py** and **4.a.py:** These scripts train a custom-built neural network for a regression task. The input is expected from a txt file located in a folder called Dane (the dataset).
- **4.b.I.py:** A regression model built using Keras, including early stopping to prevent overfitting. The model uses two hidden layers.
- **4.b.II.py:** A binary classification task (Moons dataset) using Keras, with visualization of decision boundaries.

## Key Results and Insights
### Activation Functions: Sigmoid vs ReLU
- 4.a.fa.py (Sigmoid Activation): The network tends to saturate for large values and struggles with gradient issues. The MSE and R² plots show slower convergence compared to ReLU.
- 4.a.py (ReLU Activation): The network shows faster convergence in both MSE and R² metrics. The ReLU activation function avoids the vanishing gradient problem seen with Sigmoid and provides more accurate predictions, as demonstrated in the test vs. prediction scatter plots.

### Keras-Based Models (Regression and Classification)
- 4.b.I.py (Keras Regression): With the addition of early stopping, the model reaches its optimal performance early, preventing overfitting. The MSE and R² values show significant improvement over custom-built models. The plots reflect smoother convergence with validation data.
- 4.b.II.py (Keras Classification): The neural network successfully learns the decision boundary of the moon-shaped data. The accuracy plots show strong performance, with decision boundary visualizations reflecting how well the model separates the classes.

## Visualizations
Each script generates visualizations that track the model’s performance over time, including:

- Mean Squared Error (MSE) vs. Epochs: Demonstrates how the error decreases during training.
- R² Score vs. Epochs: Shows the improvement in fit quality.
- Data Scatter Plots with Predictions: Compares the true data against predictions, showing how well the model generalizes.
- Classification Decision Boundaries: In the classification model, the decision boundary shows how well the model classifies the dataset.

![DataSet_3](https://github.com/user-attachments/assets/61d19f86-9576-4229-b5d8-bee6268e86b8)
![DataSet_9](https://github.com/user-attachments/assets/f2ebbbc8-57ab-4325-a351-0259f9ce9182)
![DataSet_4](https://github.com/user-attachments/assets/0982012b-a3aa-4302-a483-7f69082b6635)
![DataSet_7](https://github.com/user-attachments/assets/f0726f8d-0179-4842-833b-4970aca5b79a)

## Conclusions
From these visualizations, we can conclude the following:

1. Activation Functions: ReLU generally performs better than Sigmoid for regression tasks in terms of speed and accuracy.
2. Early Stopping: The use of early stopping in the Keras model prevents overfitting and ensures the model does not train for unnecessary epochs.
3. Binary Classification: Keras efficiently handles classification tasks with a relatively simple network, providing clear decision boundaries and high accuracy.

## What I Learned
- **Activation Functions Matter:** The choice of activation function significantly affects the training dynamics of neural networks. Sigmoid suffers from vanishing gradients, while ReLU is much more efficient for deeper models.
- **Custom vs. Framework-Based Models:** Writing neural networks from scratch is a great learning experience, but using frameworks like Keras simplifies development and optimization, allowing for more complex architectures with less code.
- **Early Stopping and Optimization:** Incorporating callbacks such as early stopping helps in achieving better model performance and prevents overfitting, especially when dealing with noisy or smaller datasets.
- **Visualization is Key:** Visualizing the training process is crucial to understanding how well the model is learning and identifying potential issues such as overfitting or underfitting.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
