#!/usr/bin/env python
# coding: utf-8

# # 🌟 Assignment 2. Operation "AI GeoGuessr"
# 
# ### Deadline: Friday, October 24, 2025, 11:59:00 PM CET (GMT +2)
# 
# ## 🎯 Mission Overview
# 
# The operation consists of two independent missions. Similar to the previous operation, students are required to solve the tasks by achieving the required score and answering the follow-up questions.
# 
# ## 🧪 Neural Networks and Unsupervised Learning
# 
# 1. **Mission 2.1 (Supervised Learning)**: 
#    - Solve a supervised learning problem using a multilayer perceptron (MLP).
# 
# 2. **Mission 2.2 (Unsupervised Learning)**:
#    - Given an unlabeled dataset, your task is to cluster similar data points and achieve 100% clustering accuracy. You will not have access to the true labels, but you can verify your cluster assignments using the Kaggle competition provided.
# 
# ## 📝 Delivery Format (Blackboard)
# 
# Please submit **two files**:
# 1. The completed Jupyter notebook.
# 2. The solution for the second dataset as a CSV file with two `int` columns: `['ID', 'cluster']`.
# 
# ## Kaggle details
# - Invitation link: https://www.kaggle.com/t/dfb72837bdb346449768b8f6ce50b6dc
# - Kaggle is a popular platform for data science competitions.
# - It allows us to create a Private competition where you can submit your solutions and verify whether you are thinking in the right direction.
# - The results of the competition is visible only to the competition participants. 
# - We will not grade the submissions on Kaggle, we set it up to let you check your clustering.
# - You still have to deliver the `.csv` file of the solution and the complete `.ipynb` notebook with discussions and solution code.  
# 
# > Good luck, comrade!

# # 🧠 Mission 2.1: Decoding SkyNet's Neural Encryption **(using Neural Networks)**
# 
# ### 🌐 The Discovery
# The dataset consists of the same "Synapse Cipher" data from Assignment 1.
# 
# ### 🎯 Your Mission
# 1. Implement a multilayer perceptron (MLP) using one of the following frameworks: Keras, PyTorch, or TensorFlow.
# 2. Solve the non-linear problem without manual feature engineering.
# 3. Predict SkyNet's binary decisions (0 or 1) based on paired signals.
# 4. Improve performance by using techniques such as learning rate scheduling, selecting a good optimizer, and fine-tuning hyperparameters.
# 
# > Note: There are no restrictions on the tricks you can use — Have fun :) 
# 
# ### 📊 Formal Requirements
# 1. **Implementation**:
#    - Develop a Neural Network using predefined functions/layers.
#    - Use one of the popular frameworks: Keras, PyTorch, or TensorFlow.
#    - Implement a manual learning rate scheduler with warmup and cosine decay.
# 
# 2. **Performance**: Achieve at least **0.92** accuracy on the test set.
# 
# 3. **Discussion**:
#    - How can you make sure the results are reproduable?
#    - Visualize the network's architecture and decision boundary.
#    - Which optimizer did you choose? Discuss the differences between SGD and Adam.
#    - Plot the learning rate curve. Did the learning rate scheduling improve performance? Why or why not?
#    - Conduct a simple ablation study of each architectural and optimization choice concerning test accuracy.

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('nn_data.csv')
train = data[data['split'] == 'train']
test = data[data['split'] == 'test']


# In[3]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.experimental import enable_op_determinism
from tensorflow.keras.callbacks import LearningRateScheduler

from keras import ops
import keras

from sklearn.model_selection import train_test_split

set_random_seed(1)
enable_op_determinism()


# In[4]:


y = train[['y']]
X = train.drop(columns=['y', 'split'])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_test = test[['y']]
X_test = test.drop(columns=['y', 'split'])


# print(
#     train.describe(),'\n',
#     test.describe()
# )

# In[5]:


model = Sequential()
model.add(Dense(10, activation='relu', name='layer1'))
model.add(Dense(5, activation='relu', name='layer2'))
model.add(Dense(1, activation='sigmoid', name='output'))

model(X_train)


# model.summary()

# In[6]:


model.compile(
    optimizer=SGD(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.BinaryAccuracy()],
)


# In[7]:


EPOCHS = 100


# In[8]:


def learning_rate_scheduler(lr0, warmup_steps):
    def cosine_decay(epoch):
        if epoch < warmup_steps:
            return lr0 * (epoch + 1) / warmup_steps
        else:
            return 0.5 * lr0 * (1 + np.cos(np.pi * ((epoch - warmup_steps) / (EPOCHS - warmup_steps))))
    return cosine_decay

lr_scheduler = learning_rate_scheduler(lr0=0.1, warmup_steps=10)


# In[9]:


history = model.fit(
    X_train,
    y_train,
    batch_size=16,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[LearningRateScheduler(lr_scheduler)],
    verbose=1
)


# In[10]:


train_losses = history.history['loss']
val_losses = history.history['val_loss']
x_values = np.arange(1, len(train_losses)+1)
plt.plot(x_values, train_losses, label='train loss')
plt.plot(x_values, val_losses, label='validation loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()


# In[11]:


loss, accuracy = model.evaluate(X_train, y_train)
print("Accuracy:", accuracy)


# In[12]:


epochs = np.arange(EPOCHS)
lrs = [lr_scheduler(epoch) for epoch in epochs]

plt.plot(epochs, lrs, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Mjau Mjau')
plt.grid(True)
plt.legend()
plt.show()


# In[13]:


plt.figure(figsize=(6, 4))
plt.scatter(train['x0'], train['x1'], c=train['y'], cmap='coolwarm')
plt.grid(True)
plt.xlabel('x0', fontsize=14)
plt.ylabel('x1', fontsize=14)
plt.title('x1 vs. x0', fontsize=16)
plt.colorbar()
plt.show()


# In[14]:


print("Stopping here.")
assert False, "Execution interrupted on purpose."


# # 📶📡🛰️🤳 Mission 2.2: Locate the AI Servers
# 
# ### 🌐 The Discovery
# The NTNU Intelligence team has intercepted encrypted documentation that potentially reveals the locations of AI-compromised servers within Trondheim. Our analysts believe the data contains information about these servers' locations. 
# 
# However, the number of servers and their exact positions remain unknown. To execute a strategic response, we need to understand the intercepted data and pinpoint the main AI servers.
# 
# We have also discovered a vulnerability in the AI's security: it has deployed a clustering verification feature on Kaggle. We can exploit this backdoor to validate our clustering hypotheses.
# 
# ### 🎯 Your Mission
# Identify the number of hidden servers (clusters) in the dataset. Assign cluster labels to each row in the original dataset and validate your clustering solution on Kaggle.  When achieved the perfect clustering accuracy, find the precise locations of these servers (clusters).
# 
# ### 🧠 The Challenge
# 1. **Large Dataset**: The intercepted data file contains numerous features, making direct manipulation and clustering difficult. Applying dimensionality reduction techniques could help reveal patterns.
# 2. **Unsupervised Learning**: No target labels are provided, so you must identify patterns based solely on the dataset's features.
# 3. **Data Obfuscation**: The AI has obfuscated the data, making it difficult to discern its true structure. You must effectively manipulate the data to visually identify server locations.
# 
# 
# ### 🚀 Your Tools
# - The intercepted AI dataset.
# - Skills in feature engineering, data manipulation, and visualization.
# - Clustering algorithms (e.g., KMeans, DBSCAN, Birch, Agglomerative Clustering, etc.) — [Clustering in Scikit-Learn](https://scikit-learn.org/stable/modules/clustering.html)
# - Dimensionality reduction techniques (e.g., PCA, t-SNE, Isomap, UMAP, etc.)
# - Kaggle competition platform with a daily submission limit of 50 attempts.
# 
# > Note: Kaggle competition link — [Kaggle Competition](https://www.kaggle.com/t/dfb72837bdb346449768b8f6ce50b6dc)
# 
# ### 📊 Formal Requirements
# 1. **Achieve 100% Clustering Accuracy** on Kaggle.
# > Note: The only way to check whether your clustering is correct is to submit it to Kaggle. We do not give you the clusters directly.
# 
# 2. **Discussion Questions**:
#    - **Dimensionality Reduction**: Which dimensionality reduction methods helped you determine the correct number of clusters? Why did some methods work better than others? Explain the differences between PCA, t-SNE, and UMAP.
#    - **Clustering Approach**: Which clustering algorithm and hyperparameters did you use? Discuss the differences between algorithms like KMeans and DBSCAN.
#    - **Data Type Analysis**: What is the hidden data type in the dataset? How does this information influence feature extraction for clustering? Can it explain why some dimensionality reduction algorithms are more effective?
#    - **Server Locations**: Identify the server locations. List the specific facilities in Trondheim and explain how you deduced their locations.
#    - **Advanced Task (Optional)**: Extract features using modern pre-trained neural networks for this data type. Apply dimensionality reduction and clustering algorithms to the extracted features.

# In[ ]:


# Hmmmmm, why is the first row skipped?
data = pd.read_csv('unsupervised_data.csv', skiprows=1, header=None)
# Get the ID column
data.reset_index(drop=False, inplace=True)
data.rename(columns={'index': 'ID'}, inplace=True)
data


# In[ ]:


# The dataset is heavy. Applying clustering directly on the dataset is likely not feasible. 
data.info(0)


# In[ ]:


# Features seem to be similarly distributed...
plt.hist(data.iloc[:, 2], bins=50, color='blue', alpha=0.5, label='Feature #2')
plt.hist(data.iloc[:, 13021], bins=50, color='red', alpha=0.5, label='Feature #13021')
plt.legend()
plt.show()


# In[ ]:


# To check your clustering, you need to assign the predicted cluster ids and submit it as a CSV file. The submission should be a CSV file with two columns: ID and cluster. 
# The ID column should contain the ID of the data point, and the cluster column should contain the cluster ID that the data point belongs to. 
# The cluster ID should be an integer. Current cluster IDs in sample_submission.csv are randomly generated.
sample_submission = pd.read_csv('sample_submission.csv')
sample_submission


# 
# ---
# 
# ## 🎯 Note: Clustering Accuracy Evaluation
# 
# The clustering accuracy metric evaluates how well the predicted clusters match the true clusters, irrespective of the specific labels or names assigned to the clusters.
# 
# This means that the evaluation is solely based on the correct grouping of data points rather than the numerical labels of the clusters themselves.
# 
# ## Key Characteristics
# 
# - **Name-Independent**: The metric cares only about how well the data points are grouped together, not the specific numerical or categorical labels used for the clusters.
# - **Focus on Grouping**: The evaluation rewards cluster assignments that correctly group the same data points together, regardless of the specific labels assigned.
# 
# ### Example
# 
# If the true cluster assignments are:
# 
# ```
# [0, 0, 0, 1, 1, 1]
# ```
# 
# and your predicted cluster assignments are:
# 
# ```
# [1, 1, 1, 0, 0, 0]
# ```
# 
# the accuracy will still be **1.0** because the grouping of points is identical, even though the numerical labels are swapped.
# 
# ## How the Metric is Computed
# 
# 1. **Contingency Matrix**: 
#    - Construct a contingency matrix that represents the overlap between the true clusters and the predicted clusters.
# 
# 2. **Optimal Correspondence**: 
#    - Use an optimization algorithm, such as the Hungarian method (linear sum assignment), to find the best possible correspondence between true and predicted labels, maximizing the number of correctly assigned data points.
# 
# 3. **Accuracy Calculation**: 
#    - Calculate the accuracy as the ratio of correctly matched data points to the total number of data points.
# 
# > This approach ensures that the evaluation is based on **cluster completeness** and **homogeneity**, rewarding cluster assignments that correctly group the same data points together, regardless of the specific labels used.
# 
# ---

# In[ ]:


import numpy as np
import scipy.optimize
import sklearn.metrics

def calculate_clustering_accuracy(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    """
    Calculate the clustering accuracy between true labels and cluster labels.

    The function constructs a contingency matrix between the true labels and cluster labels.
    It then uses the Hungarian algorithm (also known as linear sum assignment) to find the
    best matching between the cluster labels and true labels. The clustering accuracy is 
    calculated as the number of correctly matched labels divided by the total number of labels.

    Args:
        true_labels (np.ndarray): An array of true labels for each data point.
        cluster_labels (np.ndarray): An array of cluster labels predicted by a clustering algorithm.

    Returns:
        float: The clustering accuracy, a value between 0 and 1 representing the proportion of 
               correctly matched labels.

    Example:
        >>> true_labels = np.array([0, 1, 2, 0, 1, 2])
        >>> cluster_labels = np.array([1, 2, 0, 1, 2, 0])
        >>> calculate_clustering_accuracy(true_labels, cluster_labels)
        1.0

    Raises:
        ValueError: If true_labels and cluster_labels are not of the same length.
    """
    # Check if the input labels are of the same length
    if true_labels.size != cluster_labels.size:
        raise ValueError("true_labels and cluster_labels must have the same length.")

    # Construct a contingency matrix where each cell [i, j] indicates the number of points with 
    # true label i and cluster label j.
    contingency_matrix = sklearn.metrics.cluster.contingency_matrix(true_labels, cluster_labels)

    # Find the best matching between true labels and cluster labels using the Hungarian algorithm.
    # We negate the contingency matrix because linear_sum_assignment finds the minimum cost assignment.
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-contingency_matrix)

    # Calculate the number of correctly assigned labels according to the optimal matching.
    correct_pairs = contingency_matrix[row_ind, col_ind].sum()

    # Compute the clustering accuracy as the ratio of correctly matched labels to total labels.
    accuracy = correct_pairs / true_labels.size

    return accuracy


# In[ ]:


true_labels = np.array([0, 1, 2, 0, 1, 2])
cluster_labels = np.array([1, 2, 0, 1, 2, 0])
calculate_clustering_accuracy(true_labels, cluster_labels)


# In[ ]:


true_labels = np.array([0, 0, 0, 0, 0, 0])
cluster_labels = np.array([1, 2, 0, 1, 2, 0])
calculate_clustering_accuracy(true_labels, cluster_labels)


# In[ ]:


true_labels = np.array([1, 1, 1, 2, 2, 2])
cluster_labels = np.array([0, 0, 0, 0, 0, 0])
calculate_clustering_accuracy(true_labels, cluster_labels)

