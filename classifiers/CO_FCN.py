import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder as OHE

from utils.custom_filters import increasing_trend_filter, decreasing_trend_filter, peak_filter

# Allow for memory growth in your GPU

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class COFCN:

    def __init__(self, output_directory, length_TS, n_classes, n_filters=128, batch_size=16, epochs=2000, max_cf_length=6):

        '''
        Class defining the Custom Only Fully Convolutional Network (CO-FCN)

        Args:

            output_directory : the directory to save all the trained models and metric plots (str)
            length_TS: Length of the input Time Series (int)
            n_filters : number of filters to learn in the learnable part of the model (int)
            n_classes : number of classes in the dataset (int)
            batch_size : number of samples in each batch (int)
            epochs : number of epochs to train the model (int)
            max_cf_length : maximum number of custom filters lengths to use per type (inc, dec, peak) (int)
        '''

        self.output_directory = output_directory
        self.length_TS = length_TS
        self.n_classes = n_classes

        self.n_filters = n_filters
        self.batch_size = batch_size
        self.epochs = epochs

        self.kernel_sizes = [5, 3] # as defined for the second and third layer of the FCN model in Wang et al.

        self.max_cf_length = max_cf_length

        self.increasing_trend_kernels = [2**i for i in range(1,self.max_cf_length + 1)]
        self.decreasing_trend_kernels = [2**i for i in range(1,self.max_cf_length + 1)]
        self.peak_kernels = [2**i for i in range(1,self.max_cf_length + 1)]

        self.build_model()
    
    def build_model(self):

        input_shape = (self.length_TS, len(self.increasing_trend_kernels) + len(self.decreasing_trend_kernels) + len(self.peak_kernels))

        self.input_layer = tf.keras.layers.Input(input_shape)
        
        self.conv1 = tf.keras.layers.Conv1D(filters=2 * self.n_filters,kernel_size=self.kernel_sizes[0],padding='same',strides=1)(self.input_layer)
        self.conv1 = tf.keras.layers.BatchNormalization()(self.conv1)
        self.conv1 = tf.keras.layers.Activation(activation='relu')(self.conv1)

        self.conv2 = tf.keras.layers.Conv1D(filters=self.n_filters,kernel_size=self.kernel_sizes[1],padding='same',strides=1)(self.conv1)
        self.conv2 = tf.keras.layers.BatchNormalization()(self.conv2)
        self.conv2 = tf.keras.layers.Activation(activation='relu')(self.conv2)

        self.gap = tf.keras.layers.GlobalAveragePooling1D()(self.conv2)

        self.output_layer = tf.keras.layers.Dense(units=self.n_classes,activation='softmax')(self.gap)

        self.model = tf.keras.models.Model(inputs=self.input_layer,outputs=self.output_layer)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.5,patience=50,
                                                         min_lr=1e-4)
        
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.output_directory+'best_model.hdf5',
                                                              monitor='loss',save_best_only=True)
        
        self.callbacks = [reduce_lr,model_checkpoint]

        self.model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

    def transform(self, x):

        n = int(x.shape[0])
        l = int(x.shape[1])

        #get the output number of channels needed
        m = len(self.increasing_trend_kernels) + len(self.decreasing_trend_kernels) + len(self.peak_kernels)

        x_transformed = np.zeros(shape=(n, l, m)) # define the transformed input ndarray

        i = 0

        # apply transformations on increasing detection filters

        for kernel_size in self.increasing_trend_kernels:

            model = increasing_trend_filter(f=kernel_size,input_length=l)
            x_transformed[:,:,i] = np.asarray(model(x)).reshape(n,l)

            i += 1

        # apply transformations on decreasing detection filters

        for kernel_size in self.decreasing_trend_kernels:

            model = decreasing_trend_filter(f=kernel_size,input_length=l)
            x_transformed[:,:,i] = np.asarray(model(x)).reshape(n,l)

            i += 1
                
        # apply transformations on peak detection filters

        for kernel_size in self.peak_kernels:

            model = peak_filter(f=kernel_size,input_length=l)
            x_transformed[:,:,i] = np.asarray(model(x)).reshape(n,l)

            i += 1

        return x_transformed

    def fit(self, xtrain, ytrain, xval=None, yval=None, plot_test=False):

        # xval and yval are only used to visualize the losses and accuracies and not for training.

        n = int(xtrain.shape[0])

        mini_batch_size = min(self.batch_size,n // 10)

        ohe = OHE(sparse=False)
        ytrain = np.expand_dims(ytrain,axis=1)
        ytrain = ohe.fit_transform(ytrain)

        xtrain_transformed = self.transform(x=xtrain)

        if plot_test:

            xval_transformed = self.transform(x=xval)

            ohe = OHE(sparse=False)
            yval = np.expand_dims(yval, axis=1)
            yval = ohe.fit_transform(yval)

        if plot_test:

            hist = self.model.fit(xtrain_transformed, ytrain, batch_size=mini_batch_size, epochs=self.epochs,
                                  callbacks=self.callbacks, validation_data=(xval_transformed, yval))

        else:

            hist = self.model.fit(xtrain_transformed, ytrain, batch_size=mini_batch_size, epochs=self.epochs,
                                  callbacks=self.callbacks)

        plt.figure(figsize=(20,10))

        plt.plot(hist.history['loss'], lw=3, color='blue', label="Training Loss")

        if plot_test:
            plt.plot(hist.history['val_loss'], lw=3, color='red', label="Validation Loss")

        plt.savefig(self.output_directory+'loss.pdf')
        plt.cla()

        plt.plot(hist.history['accuracy'], lw=3, color='blue', label="Training Accuracy")

        if plot_test:
            plt.plot(hist.history['val_accuracy'], lw=3, color='red', label="Validation Accuracy")
        
        plt.savefig(self.output_directory + 'accuracy.pdf')

        plt.cla()
        plt.clf()

    def predict(self, xtest, ytest):

        model = tf.keras.models.load_model(self.output_directory+'best_model.hdf5',compile=False)

        xtest_transformed = self.transform(x=xtest)

        ypred = model.predict(xtest_transformed)
        ypred = np.argmax(ypred,axis=1)

        tf.keras.backend.clear_session()

        return accuracy_score(y_true=ytest,y_pred=ypred,normalize=True)