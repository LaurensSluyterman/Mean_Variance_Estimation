import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# Load all the faces and ages
faces = np.load('./UTKFaceData/images.npy')
ages = np.load('./UTKFaceData/ages.npy')

# Filter out older than 80
indices = np.where(ages < 80)
X_2 = faces[indices]
Y_2 = ages[indices]

# Filter out 2/3 of pictures with age 1.
indices2 = np.where(Y_2 == 1)
indices21 = indices2[0][0:len(indices2[0]) // 3]  # Take the first 1/3
indices_final = np.hstack((np.where(Y_2 > 1)[0], indices21))  # Combine with all other ages
X_3 = X_2[indices_final]
Y_3 = Y_2[indices_final]

# resize to 64x64
X_resized = np.zeros((len(Y_3), 64, 64, 3))
for i in range(0, len(X_3)):
    if i%100 == 0:
        print(i)
    X_resized[i] = resize(X_3[i], (64, 64, 3))


# Save the final data
np.save('./UTKFaceData/images_resized_2.npy', X_resized)
np.save('./UTKFaceData/ages_resized_2.npy', Y_3)

# Visual check
for i in range(0, 10):
    plt.title(f'{Y_3[i]}')
    plt.imshow(X_resized[i])
    plt.axis('off')
    plt.show()
