# Neural Network Charity Analysis: Alphabet Soup

## Background
From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. 

Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Analysis Deliverables
This analysis consists of three technical analysis deliverables and a written report including the following: 

- Deliverable 1: Preprocessing Data for a Neural Network Model
- Deliverable 2: Compile, Train, and Evaluate the Model
- Deliverable 3: Optimize the Model
- Deliverable 4: A Written Report on the Neural Network Model (README.md)

## Overview of the Analysis: 
The purpose of this analysis was to utilize a neural network machine learning model (i.e. binary classifier) to predict whether or not applicants would be successful if funded by Alphabet Soup. The goal of this analysis was to help Alphabet Soup's business identify the optimal funding candidates. 

We utilized the dataset to identify our target (IS_SUCCESSFUL) and features (multiple, see code) to train our model and make our predictions. 

## Results: 

The results of our analysis is outlined below. 

- **Data Preprocessing**
  - What variable(s) are considered the **target(s)** for your model? 
    - IS_SUCCESSFUL (Boolean - Yes/No)
  - What variable(s) are considered to be the **features** for your model?
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE-CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT
  - What variable(s) are **neither targets nor features**, and should be removed from the input data?
    - EIN
    - NAME

- **Compiling, Training, and Evaluating the Model**
  - How many **neurons, layers, and activation functions** did you select for your neural network model, and why?
    - On my third, and most successful, optimization attempt, I utilized **45 input features (columns)**, **one combined input/hidden layer with 20 neurons**, and the **output layer**. 
    - I utilized the **"relu" activation function on my first layer** because I wanted to identify and train on nonlinear relationships in the dataset. 
    - I utilized the **"sigmoid" activation function on my output layer** because our model was a classification model concerned with a yes or no binary decision (IS_SUCCESSFUL). In this case, I utilized a sigmoid activation function to produce a probability output.  
    - The Keras module does not have specific classes for input, hidden, and output layers. We built the layers using the Dense class with the input and first hidden layer in the same instance. 
    - We utilized the **"Adam" optimization function** to shape and mold our model while it was being trained. One benefit of the "Adam" optimizer is that it should prevent the algorithm from getting stuck on weaker classifying variables and features. 
    - We utilized the **"binary_crossentropy" loss function** since it was specifically designed to evaluate a binary classification model such as this. 
```
# Define the model - deep neural net
# i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train[0])
hidden_nodes_layer1 = 20

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_2 (Dense)             (None, 20)                900       
                                                                 
 dense_3 (Dense)             (None, 1)                 21        
                                                                 
=================================================================
Total params: 921
Trainable params: 921
Non-trainable params: 0
_________________________________________________________________
```

  - **Were you able to achieve the target model performance?**
    - I was not able to achieve the target model performance of 75% accuracy, but I was close on my 3rd optimization attempt (see code). 
    - I achieved approximately 69% accuracy on my final optimization attempt. 
    
```
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

```
268/268 - 0s - loss: 4.8151 - accuracy: 0.6894 - 213ms/epoch - 795us/step
Loss: 4.815110683441162, Accuracy: 0.6894460916519165
```

  - **What steps did you take to try and increase model performance?**
     - added neurons to hidden layer(s) -- **Optimization2, Optimization3**
     - added additional hidden layer(s) -- **Optimization1, Optimization2**
     - increased and reduced epochs to the training process -- **Optimization1**
     - removed additional columns to attempt to reduce "noise" in the model (i.e. USE_CASE, SPECIAL_CONSIDERATIONS) - **Optimization1**
     - experimented with the activation functions used in the layers -- **Optimization1, Optimization2, Optimization3**

## Summary: 

- Overall, my third Optimization of the deep learning neural network model was the most successful of all my attempts. Interestingly, this was arguably the most simple of the models with only 1 combined input/hidden layer. This supports the statement I read in several online sources and in class: most problems can be solved with only one hidden layer. Adding complexity to the layers did not translate into a more accurate model. I did increase the number of neurons in the hidden layer to 20 in my most successful attempt. With additional time and tweaks, I could have probably improved the model even more, but this exercise did seem to illustrate that the accuracy seems to plateau despite additional optimization efforts. 
- My recommendation would be to investigate whether a **Support Vector Machine (SVM) model** could do with our classification problem. SVMs are another type of binary classifier that can evaluate multiple data types. SVMs are less prone to overfitting because they don't try to gather all data within a boundary as with neural networks. SVMs have their disadvantages as well and can miss critical features; however, they can outperform the basic neural network in straightforward binary classification problems such as this. 
