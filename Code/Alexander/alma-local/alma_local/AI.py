
from __future__ import print_function
import keras
from keras import utils as np_utils
import tensorflow
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

"""Hyper-parameters data-loading and formatting"""
batch_size = 128
num_classes = 10
epochs = 10

img_rows, img_cols = 28, 28

(x_train, lbl_train), (x_test, lbl_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

"""Preprocessing"""
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
pk = x_train

x_train /= 255
x_test /= 255

y_train = keras.utils.np_utils.to_categorical(lbl_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(lbl_test, num_classes)


# """Define model"""
# model = Sequential()

# model.add(Flatten())
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(64, activation = 'relu'))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#                optimizer=tensorflow.keras.optimizers.SGD(learning_rate = 0.1),
#         metrics=['accuracy'],)

# fit_info = model.fit(x_train, y_train,
#            batch_size=batch_size,
#            epochs=epochs,
#            verbose=1,
#            validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)

# print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))
# print("\n\n")
# # print(model.summary())


# def plotter(fit_info, epochs):
#     accuracy = fit_info.history['accuracy']  # Training accuray
#     val_acc = fit_info.history['val_accuracy']  # Validation accuracy
#     # loss = fit_info.history['loss']  # Training loss
#     # val_loss = fit_info.history['val_loss']  # Validation loss

#     epochs_lc = [i+1 for i in range(epochs)]

#     plt.plot(epochs_lc, accuracy, color='r',  marker='o', linestyle='--', label='Training accuracy')
#     plt.plot(epochs_lc, val_acc, color='b', marker='o', linestyle='--', label='Validation accuracy')
#     plt.xticks(epochs_lc)
#     plt.title('Training and validation accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     # plt.figure()
#     # plt.plot(epochs, loss, 'bo', label='Training loss')
#     # plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     # plt.title('Training and validation loss')
#     # plt.legend()

#     plt.savefig('accuracy_cleaned.png', bbox_inches='tight')
#     plt.show()

# """Question 2d"""
# epochs = 40;
# # Regularization factors from 0.000001 to 0.001
# reg_factors = np.linspace(0.000001,0.001,5)
# print(reg_factors)
# info = []

# for i in range(len(reg_factors)):
#     for j in range(3):
#         print(reg_factors[i])
#         model = Sequential()

#         model.add(Flatten())
#         model.add(Dense(500, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(reg_factors[i])))
#         model.add(Dense(300, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(reg_factors[i])))
#         model.add(Dense(num_classes, activation='softmax'))

#         model.compile(loss=keras.losses.categorical_crossentropy,
#                        optimizer=tensorflow.keras.optimizers.SGD(learning_rate = 0.1),
#                 metrics=['accuracy'],)

#         fit_info = model.fit(x_train, y_train,
#                    batch_size=batch_size,
#                    epochs=epochs,
#                    verbose='0',
#                    validation_data=(x_test, y_test))
#         score = model.evaluate(x_test, y_test, verbose=0)

#         print('Test loss: {}, Test accuracy {}'.format(score[0], score[1]))
#         print("\n\n")
#         info.append([score[1], i, j])

# def plotter(fit_info):
#     factors=[0.000001, 0.00025075, 0.0005005, 0.00075025, 0.001]
#     mean = []
#     std = []
#     for _ in range(0, len(fit_info), 3):
#         tmp = []
#         for i in range(3):
#             tmp.append(fit_info[i][0])
#         mean.append(np.mean(tmp, axis=0))
#         std.append(np.std(tmp, axis=0))

#     sel=[fit_info[x][0] for x in range(0,len(fit_info),3)]
#     plt.plot(factors, sel, label="Validation accuracy", c='r')
#     plt.plot(factors, [sel[i]-std[i] for i in range(len(std))], label="Lower", c='g', ls='--')
#     plt.plot(factors, [sel[i]+std[i] for i in range(len(std))], label="Upper", c='g', ls='--')
#     plt.xlabel('Regularization factors')
#     plt.ylabel('Accuracy')

# """Question 3a"""
# epochs = 40
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(500, activation='relu'))
# model.add(Dense(300, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.SGD(learning_rate=0.1),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test accuracy:', score[1])

noise_level = 0.5
def salt_and_pepper(input, noise_level=noise_level):
    """
    This applies salt and pepper noise to the input tensor - randomly setting bits to 1 or 0.
    Parameters
    ----------
    input : tensor
        The tensor to apply salt and pepper noise to.
    noise_level : float
        The amount of salt and pepper noise to add.
    Returns
    -------
    tensor
        Tensor with salt and pepper noise applied.
    """
    # salt and pepper noise
    a = np.random.binomial(size=input.shape, n=1, p=(1 - noise_level))
    b = np.random.binomial(size=input.shape, n=1, p=0.5)
    c = (a==0) * b
    return input * a + c

#data preparation
flattened_x_train = x_train.reshape(-1,784)
flattened_x_train_seasoned = salt_and_pepper(flattened_x_train, noise_level=noise_level)

flattened_x_test = x_test.reshape(-1,784)
flattened_x_test_seasoneed = salt_and_pepper(flattened_x_test, noise_level=noise_level)

latent_dim = 96  

input_image = keras.Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_image)
encoded = Dense(latent_dim, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = keras.Model(input_image, decoded)
encoder_only = keras.Model(input_image, encoded)

encoded_input = keras.Input(shape=(latent_dim,))
decoder_layer = Sequential(autoencoder.layers[-2:])
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

fit_info_AE = autoencoder.fit(flattened_x_train_seasoned, flattened_x_train,
                epochs=32,
                batch_size=64,
                shuffle=True,
                validation_data=(flattened_x_test_seasoneed, flattened_x_test))

noises = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
num_rows = 6
num_cols = 10
f, ax = plt.subplots(nrows = num_rows , ncols = num_cols, figsize = (30, 30)) 
for row in range(num_rows):
   for col in range(num_cols):
    ax[0,col].imshow(tf.reshape(salt_and_pepper(flattened_x_train[10].reshape(1,-1), noise_level = noises[col]),(28, 28)))
    ax[1,col].imshow(tf.reshape(autoencoder(salt_and_pepper(flattened_x_train[10].reshape(1,-1),noise_level = noises[col])), shape = (28, 28)))

    ax[2,col].imshow(tf.reshape(salt_and_pepper(flattened_x_train[100].reshape(1,-1), noise_level = noises[col]),(28, 28)))
    ax[3,col].imshow(tf.reshape(autoencoder(salt_and_pepper(flattened_x_train[100].reshape(1,-1),noise_level = noises[col])), shape = (28, 28)))

    ax[4,col].imshow(tf.reshape(salt_and_pepper(flattened_x_train[1000].reshape(1,-1), noise_level = noises[col]),(28, 28)))
    ax[5,col].imshow(tf.reshape(autoencoder(salt_and_pepper(flattened_x_train[1000].reshape(1,-1),noise_level = noises[col])), shape = (28, 28)))


plt.savefig('4b_plots', bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    pass
    # plotter(info)

    # plt.savefig('2d.png', bbox_inches='tight')
    # plt.show()
