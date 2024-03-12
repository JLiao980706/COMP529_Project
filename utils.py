import numpy as np
import torch


def assign_sub_matrices(matrix_part1, matrix_part2, ranks):
    rand_indices = np.arange(ranks.sum())
    np.random.shuffle(rand_indices)
    sub_mats = []
    for idx in range(ranks.shape[0]):
        indices = rand_indices[ranks[:idx].sum():ranks[:idx].sum() + ranks[idx]]
        sub_mats.append((matrix_part1[:,indices], matrix_part2[indices]))
    return sub_mats


def reassign_sub_matrices(submatrices):
    matrices_part1 = np.concatenate([a.cpu().numpy() for a, _ in submatrices], axis=1)
    matrices_part2 = np.concatenate([b.cpu().numpy() for _, b in submatrices], axis=0)
    ranks = np.array([a.shape[1] for a, _ in submatrices])
    return assign_sub_matrices(matrices_part1, matrices_part2, ranks), (matrices_part1, matrices_part2)


def relu(x):
    return np.maximum(x, 0)


def generate_target_func(input_dim, hidden_dim, output_dim, rank_W1, rank_W2):
    fixed_W1 = np.random.normal(size=(input_dim, hidden_dim)) / np.sqrt(input_dim)
    fixed_W2 = np.random.normal(size=(hidden_dim, output_dim)) / np.sqrt(output_dim + hidden_dim)

    print('Operator Norm of fixed W1: ', np.linalg.norm(fixed_W1, ord=2))
    print('Operator Norm of fixed W2: ', np.linalg.norm(fixed_W2, ord=2))

    A1 = np.random.normal(size=(input_dim, rank_W1)) / np.sqrt(input_dim)
    B1 = np.random.normal(size=(rank_W1, hidden_dim)) / np.sqrt(hidden_dim)
    A2 = np.random.normal(size=(hidden_dim, rank_W2)) / np.sqrt(hidden_dim)
    B2 = np.random.normal(size=(rank_W2, output_dim)) / np.sqrt(output_dim)

    W1 = fixed_W1 + 3 * A1 @ B1
    W2 = fixed_W2 + 3 * A2 @ B2

    print('Operator Norm of W1: ', np.linalg.norm(W1, ord=2))
    print('Operator Norm of W2: ', np.linalg.norm(W2, ord=2))
    
    return lambda X: relu(X @ W1) @ W2, fixed_W1, fixed_W2


def generate_data(input_dim, num_train, num_test, target_func):
    X = np.random.normal(size=(num_train + num_test, input_dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = target_func(X)
    return (X[:num_train], Y[:num_train]), (X[num_train:], Y[num_train:])

def data_to_tensor(data_pair):
    return torch.from_numpy(data_pair[0].astype(np.float32)), torch.from_numpy(data_pair[1].astype(np.float32))

    