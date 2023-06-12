# Matrix Factorization
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of Matrix Factorization in Python. Matrix Factorization is a collaborative filtering technique commonly used in recommender systems. It aims to factorize a user-item rating matrix into two lower-rank matrices, representing user and item embeddings. These embeddings capture the latent features of users and items and can be used to predict missing ratings and generate personalized recommendations.

## Introduction to Matrix Factorization
Matrix Factorization is a dimensionality reduction technique that decomposes a matrix into two lower-rank matrices. In the context of recommender systems, the matrix being factorized is the user-item rating matrix, where each entry represents the rating given by a user to an item. The goal is to find two matrices, one representing users and the other representing items, such that their product approximates the original rating matrix.

The factorization process discovers latent features or factors that represent the underlying characteristics of users and items. These latent factors can capture various attributes such as genre preferences, item popularity, user tastes, etc. By multiplying the user and item embeddings, we obtain an approximation of the rating matrix, which can be used to predict missing ratings and generate personalized recommendations.

## Files
- `mf.py`: This file contains the implementation of Matrix Factorization using Python. It includes functions for data encoding, creating embeddings, computing predictions, calculating the cost function, performing gradient descent, and more. The file is well-documented and provides detailed explanations for each function.

## Dataset
The implementation uses the MovieLens dataset, specifically the `ml-latest-small` dataset. You can download the dataset from the following link: [MovieLens Dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small.zip) . The dataset contains user-item ratings for movies, which will be used to train and evaluate the matrix factorization model.

## Getting Started
To get started with this repository, follow these steps:

1. Make sure you have Python installed on your system (version 3 or above).
2. Install the required dependencies by running the following command:
```bash
$ pip install -r requirements.txt
```
3. Download and extract the MovieLens dataset from the provided link, and place the extracted dataset folder (`ml-latest-small`) in the same directory as the `mf.py` file.
```bash
$ wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip
$ unzip ml-latest-small.zip
```

## Usage
To use the matrix factorization implementation in `mf.py`, you can follow these steps:

1. Import the necessary libraries and functions from `mf.py` into your Python script or interactive environment.
```python
import numpy as np
import pandas as pd
from scipy import sparse
from mf import encode_data, encode_new_data, create_embedings, gradient_descent, cost
```

2. Load the dataset using pandas and preprocess it using the provided `encode_data` and `encode_new_data` functions. 
```python
df = pd.read_csv("ml-latest-small/ratings.csv")
df, num_users, num_movies = encode_data(df)
```

3. Create initial user and item embeddings using the `create_embedings` function. 
```python
K = 10  # Number of factors in the embedding
emb_user = create_embedings(num_users, K)
emb_movie = create_embedings(num_movies, K)
```

4. Use the `gradient_descent` function to train the matrix factorization model on the training dataset. 
```python
emb_user, emb_movie = gradient_descent(df, emb_user, emb_movie, iterations=100)
```
5. Optionally, you can evaluate the model's performance on a validation dataset using the `cost` function.
```python
df_val = pd.read_csv("ml-latest-small/ratings_val.csv")
df_val = encode_new_data(df_val, df)
validation_cost = cost(df_val, emb_user, emb_movie)
print("Validation cost:", validation_cost)
```
6. After training, you can make recommendations for users by computing the dot product of their user embeddings and item embeddings.
```python
user_id = 42
user_embedding = emb_user[user_id]
item_embeddings = emb_movie
predicted_ratings = np.dot(user_embedding, item_embeddings.T)
top_movies = np.argsort(predicted_ratings)[-5:][::-1]
for movie_id in top_movies:
    print("Movie ID:", movie_id)
```

Please refer to the code in `mf.py` for more details on each function and their parameters. You can also find comments within the code that explain the purpose and functionality of each function.

## Conclusion
The Matrix Factorization repository provides a Python implementation of Matrix Factorization, a popular technique used in recommender systems. It allows you to factorize a user-item rating matrix into user and item embeddings, which can be used to predict ratings and generate personalized recommendations. By utilizing gradient descent with momentum, the repository enables efficient training and optimization of the embeddings. Feel free to explore the repository and customize the code to suit your specific needs.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
The initial codebase and project structure is adapted from the MSDS 630 course materials provided by the University of San Francisco (USFCA-MSDS). Special thanks to the course instructors for the inspiration.
