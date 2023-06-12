import numpy as np
import pandas as pd
from scipy import sparse


def proc_col(col):
    """
    Encodes a pandas column with values between 0 and n-1
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o: i for i, o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)


def encode_data(df):
    """
    Encodes rating data with continous user and movie.

    Arguments:
      df: a csv file with columns userId, movieId, rating

    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
    """
    d = {
        "userId": proc_col(df["userId"])[1],
        "movieId": proc_col(df["movieId"])[1],
        "rating": df["rating"],
    }
    df = pd.DataFrame(data=d)
    num_users = proc_col(df["userId"])[2]
    num_movies = proc_col(df["movieId"])[2]
    return df, num_users, num_movies


def encode_new_data(df_val, df_train):
    """
    Encodes df_val with the same encoding as df_train.
    
    Returns:
      df_val: dataframe with the same encoding as df_train
    """
    userId = proc_col(df_train["userId"])[0]
    movieId = proc_col(df_train["movieId"])[0]

    data = {
        "userId": df_val["userId"].apply(
            lambda x: userId[x] if x in userId else np.nan
        ),
        "movieId": df_val["movieId"].apply(
            lambda x: movieId[x] if x in movieId else np.nan
        ),
        "rating": df_val["rating"],
    }
    df_val = pd.DataFrame(data).dropna().astype("int")
    return df_val


def create_embedings(n, K):
    """
    Create a numpy random matrix of shape n, K
    The random matrix should be initialized with uniform values in (0, 6/K)
    
    Arguments:
      Inputs:
        n: number of items/users
        K: number of factors in the embeding

    Returns:
      emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6 * np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """
    Returns a sparse matrix constructed from a dataframe.
    This code assumes the df has columns: movieID,userID,rating
    """
    values = df[column_name].values
    ind_movie = df["movieId"].values
    ind_user = df["userId"].values
    return sparse.csc_matrix((values, (ind_user, ind_movie)), shape=(nrows, ncols))


def sparse_multiply(df, emb_user, emb_movie):
    """
    This function returns U*V^T element wise multi by R as a sparse matrix.
    It avoids creating the dense matrix U*V^T
    """
    df["Prediction"] = np.sum(
        emb_user[df["userId"].values] * emb_movie[df["movieId"].values], axis=1
    )
    return df2matrix(
        df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction"
    )


def cost(df, emb_user, emb_movie):
    """
    Computes Mean Square Error.

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      error(float): this is the MSE
    """
    df, num_users, num_movies = encode_data(df)

    pred = sparse_multiply(df, emb_user, emb_movie)
    Y = df2matrix(df, pred.shape[0], pred.shape[1])

    error = np.sum(np.square(pred.toarray() - Y.toarray())) / Y.nnz
    return error


def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """
    Computes finite difference on MSE(U, V).
    This function is used for testing the gradient function.
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1) / e


def gradient(df, Y, emb_user, emb_movie):
    """
    Computes the gradient.

    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]

    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies

    Returns:
      d_emb_user
      d_emb_movie
    """
    Y = np.array(Y.todense())
    mask = np.where(Y != 0, 1, 0)
    N = mask.sum()

    pred_user = (emb_user @ emb_movie.T) * mask
    pred_movie = (emb_movie @ emb_user.T).T * mask

    grad_user = -2 / N * ((Y - pred_user) @ emb_movie)
    grad_movie = -2 / N * ((Y - pred_movie).T @ emb_user)
    return grad_user, grad_movie


def gradient_descent(
    df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None
):
    """
    Computes gradient descent with momentum (0.9) for a number of iterations.

    Prints training cost and validation cost (if df_val is not None) every 50 iterations.

    Returns:
      emb_user: the trained user embedding
      emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    Y_mat = np.array(Y.todense())
    mask = np.where(Y_mat != 0, 1, 0)

    pred_user = (emb_user @ emb_movie.T) * mask
    pred_movie = (emb_movie @ emb_user.T).T * mask

    v_user = np.zeros(emb_user.shape)
    v_movie = np.zeros(emb_movie.shape)

    for i in range(iterations):
        v_user = 0.9 * v_user + 0.1 * gradient(df, Y, emb_user, emb_movie)[0]
        v_movie = 0.9 * v_movie + 0.1 * gradient(df, Y, emb_user, emb_movie)[1]

        emb_user = emb_user - learning_rate * v_user
        emb_movie = emb_movie - learning_rate * v_movie

        if i % 50 == 0:
            if df_val == None:
                print(f"{i} {cost(df, emb_user, emb_movie)} None")
            else:
                print(
                    f"{i} {cost(df, emb_user, emb_movie)} {cost(df_val, emb_user, emb_movie)}"
                )
    return emb_user, emb_movie
