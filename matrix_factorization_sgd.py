import numpy as np

num_users = 10
num_items = 15
k = 5
R = np.random.randint(0, 5, size=(num_users, num_items))
U = np.random.randn(num_users, k)
V = np.random.randn(num_items, k)

# Hyperparameters
alpha = 0.01   # learning rate
lambda_reg = 0.1  # regularization
epochs = 500

# SGD over observed ratings
for epoch in range(epochs):
    cum_error = 0
    for i in range(num_users):
        for j in range(num_items):
            if R[i, j] > 0:  # only use observed ratings
                # Predict rating
                pred = U[i, :] @ V[j, :].T
                error = R[i, j] - pred
                cum_error += error ** 2

                # Update embeddings
                U[i, :] += alpha * (error * V[j, :] - lambda_reg * U[i, :])
                V[j, :] += alpha * (error * U[i, :] - lambda_reg * V[j, :])
    if epoch == 0 or (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1} completed with cumulative error: {np.sqrt(cum_error / (num_users * num_items))}")

print(U.shape, V.shape)