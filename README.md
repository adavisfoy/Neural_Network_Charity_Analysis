# Neural Network Charity Analysis: Alphabet Soup

## Background
Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

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

## Overview of the analysis: 
The purpose of this analysis was to utilize a neural network machine learning model (i.e. binary classifier) to predict whether or not applicants would be successful if funded by Alphabet Soup. The goal of this analysis is to help Alphabet Soup's business identify the optimal funding candidates. 

We utilized the dataset to identify our target (IS_SUCCESSFUL) and features (multiple, see code) to train our model and make our predictions. 

## Results: 

Using bulleted lists and images to support your answers, address the following questions.

- Data Preprocessing
  - What variable(s) are considered the target(s) for your model? 
    - IS_SUCCESSFUL (Boolean - Yes/No)
  - What variable(s) are considered to be the features for your model?
    - APPLICATION_TYPE
    - AFFILIATION
    - CLASSIFICATION
    - USE-CASE
    - ORGANIZATION
    - STATUS
    - INCOME_AMT
    - SPECIAL_CONSIDERATIONS
    - ASK_AMT
  - What variable(s) are neither targets nor features, and should be removed from the input data?
    - EIN
    - NAME

- Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
    - On my third, and most successful, optimization attempt, I utilized 45 input features (columns), one combined input/hidden layer with 20 neurons, and the output layer. 
    - I utilized the "relu" activation function on my first layer because I wanted to identify and train on nonlinear relationships in the dataset. 
    - I utilized the "sigmoid" activation function on my output layer because our model was a classification model concerned with a yes or no binary decision (IS_SUCCESSFUL). In this case, I utilized a sigmoid activation function to produce a probability output.  
    - The Keras module does not have specific classes for input, hidden, and output layers. We built the layers using the Dense class with the input and first hidden layer in the same instance. 
    - We utilized the "Adam" optimization function to shape and mold our model while it was being trained. One benefit of the "Adam" optimizer is that it should prevent the algorithm from getting stuck on weaker classifying variables and features. 
    - We utilized the "binary_crossentropy" loss function since it was specifically designed to evaluate a binary classification model such as this. 
  - Were you able to achieve the target model performance?
   - I was not able to achieve the target model performance of 75% accuracy, but I was close on my 3rd optimization attempt (see code). 
   - I achieved approximately 69% accuracy on my final optimization attempt. 
  - What steps did you take to try and increase model performance?
    - added neurons to hidden layer
    - added additional hidden layer(s)
    - added epochs to training process
    - removed columns that may not be useful for analysis
    - experimented with the activation functions used in the layers

## Summary: 

- Summarize the overall results of the deep learning model. 
- Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
