import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import tensorflow as tf
from CNN import MVECNNNetwork
from skimage.transform import resize
from utils import average_loglikelihood

# Load the data
X = np.load('./UTKFaceData/images_resized_2.npy')
Y = np.load('./UTKFaceData/ages_resized_2.npy')
X_train, Y_train = X[0:3500], Y[0:3500]
X_test, Y_test = X[3500:4500], Y[3500:4500]
X_val, Y_val = X[4500:6000], Y[4500:6000]

#%% Test the effect of separate regularization
# Dictionaries to save the results
model_dict = {}
rmse_dict = {}
LL_dict = {}

# Model parameters
epochs = 150
architecture = np.array([20, 10, 5, 2])
batch_size = 32
reg_factors_mean = [1e-3, 1e-2, 1e-1, 0.5, 1]
reg_factors_var = [1e-3, 1e-2, 1e-1, 0.5, 1]

for reg_mean in reg_factors_mean:
    for reg_var in reg_factors_var:
        print((reg_mean, reg_var))
        np.random.seed(42)
        model = MVECNNNetwork(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val,
                              n_hidden_mean=architecture,
                              n_hidden_var=architecture, n_epochs=epochs,
                              verbose=0, reg_mean=reg_mean, reg_var=reg_var,
                              batch_size=batch_size)
        mu = model.f(X_test)
        sigma = model.sigma(X_test)
        rmse_dict[(reg_mean, reg_var)] = np.sqrt(np.mean((mu - Y_test)**2))
        LL_dict[(reg_mean, reg_var)] = average_loglikelihood(Y_test, mu, sigma)
        model_dict[(reg_mean, reg_var)] = model

# Save dictionary using pickle
with open('./UTKresults/rmse.pickle', 'wb') as handle:
    pickle.dump(rmse_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./UTKresults/LL.pickle', 'wb') as handle:
    pickle.dump(LL_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('./UTKresults/models.pickle', 'wb') as handle:
    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the dictionaries
with open('./UTKresults/rmse.pickle', 'rb') as file:
    # Load the object from the file
    rmse_dict = pickle.load(file)

with open('./UTKresults/LL.pickle', 'rb') as file:
    # Load the object from the file
    LL_dict = pickle.load(file)


# Create a heatmap for RMSE
a_values = sorted(set(key[0] for key in rmse_dict.keys()))
b_values = sorted(set(key[1] for key in rmse_dict.keys()), reverse=True)


matrix = np.zeros((len(b_values), len(a_values)))
for (a, b), value in rmse_dict.items():
    a_index = a_values.index(a)
    b_index = b_values.index(b)
    matrix[b_index, a_index] = value

plt.figure(dpi=500)
sns.heatmap(matrix, annot=True, fmt='.2f', xticklabels=a_values, yticklabels=b_values, cmap='coolwarm', cbar=True)
plt.xlabel('Regularization Mean')
plt.ylabel('Regularization Variance')
plt.title('RMSE')
plt.tight_layout()
plt.show()

# Create a heatmap for NLL
matrix2 = np.zeros((len(b_values), len(a_values)))

for (a, b), value in LL_dict.items():
    a_index = a_values.index(a)
    b_index = b_values.index(b)
    matrix2[b_index, a_index] = -value

plt.figure(dpi=500)
sns.heatmap(matrix2, annot=True, fmt='.2f', xticklabels=a_values, yticklabels=b_values, cmap='coolwarm', cbar=True)
plt.xlabel('Regularization Mean')
plt.ylabel('Regularization Variance')
plt.title('NLL')
plt.tight_layout()
plt.show()


#%% Transition experiment to visualize the warm-up
epochs = 150
f = 1/3

# With warmup
np.random.seed(30)
modelt = MVECNNNetwork(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val,
                              n_hidden_mean=architecture,
                              n_hidden_var=architecture, n_epochs=epochs,
                              verbose=1, reg_mean=1e-1, reg_var=1e-1,
                              batch_size=batch_size, warmup_fraction=f, fixed_mean=False)

# Without warmup
np.random.seed(30)
model_no_wu = MVECNNNetwork(X=X_train, Y=Y_train, X_val=X_val, Y_val=Y_val,
                              n_hidden_mean=architecture,
                              n_hidden_var=architecture, n_epochs=epochs,
                              verbose=1, reg_mean=1e-1, reg_var=1e-1,
                              batch_size=batch_size, warmup_fraction=0, fixed_mean=False)


mean_predictions_nwu = np.zeros((epochs + 1, len(X_test[0:50])))
std_predictions_nwu = np.zeros((epochs + 1, len(X_test[0:50])))
y_std = np.std(Y_train)
y_mean = np.mean(Y_train)
for i in range(0, total_time+1):
    print(i)
    location = f'./UTKFaceData/transition/nowu/{i}.hdf5'
    model = tf.keras.models.load_model(location, custom_objects={'negative_log_likelihood':lambda: 0})
    outputs = model.predict(X_test[0:50])
    means = outputs[:, 0] * y_std + y_mean
    stds = np.sqrt((np.exp(outputs[:, 1]) + 1e-6)) * y_std
    mean_predictions_nwu[i] = means
    std_predictions_nwu[i] = stds


mean_predictions = np.zeros((epochs+1, len(X_test[0:50])))
std_predictions = np.zeros((epochs+1, len(X_test[0:50])))
y_std = np.std(Y_train)
y_mean = np.mean(Y_train)
for i in range(0, epochs+1):
    warmup = int(epochs*f)
    print(i)
    location = f'./UTKFaceData/transition/wu/{i}.hdf5'
    model = tf.keras.models.load_model(location, custom_objects={'negative_log_likelihood':lambda: 0})
    outputs = model.predict(X_test[0:50])
    means = outputs[:, 0] * y_std + y_mean
    stds = np.sqrt((np.exp(outputs[:, 1]) + 1e-6)) * y_std
    mean_predictions[i] = means
    std_predictions[i] = stds

#%% Create and save the plots
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.5
for i in range(0, 50):
    plt.figure(figsize=(18,5), dpi=500)
    plt.subplot(1, 3, 1)
    plt.imshow(X_test[i])
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Warm-up')
    plt.plot([i for i in range(0, epochs+1)], mean_predictions[:, i], label=r'$\hat{\mu}$')
    plt.plot([i for i in range(0, epochs+1)], std_predictions[:, i], label=r'$\hat{\sigma}$')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Age')
    plt.axvline(x=warmup, linestyle='--')
    plt.axhline(y=Y_test[i], linestyle='--')
    plt.legend()
    plt.ylim((0, 80))
    plt.xticks([0, 25, 50, 75, 100, 125, 150])
    plt.subplot(1, 3, 3)
    plt.title('No warm-up')
    plt.plot([i for i in range(0, epochs+1)], mean_predictions_nwu[:, i], label=r'$\hat{\mu}$')
    plt.plot([i for i in range(0, epochs+1)], std_predictions_nwu[:, i], label=r'$\hat{\sigma}$')
    plt.grid()
    plt.xlabel('Epoch')
    plt.ylabel('Age')
    plt.axhline(y=Y_test[i], linestyle='--')
    plt.legend()
    plt.ylim((0, 80))
    plt.xticks([0, 25, 50, 75, 100, 125, 150])
    plt.tight_layout()
    plt.savefig(f'./plots/{i}.png')
    plt.close()
