import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import scipy.io as sio
from scipy import ndimage, misc
import os
import scipy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
#path = 'C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/imagecolor/'
path = 'C:/Users/Hamidreza/Desktop/data/imcoloraug10/'

SSS=3360
data = np.zeros((SSS,64,68,3))
zz=[]

for i in range(SSS):    
    zz.append(str(i)+".jpg")

for ii, imagee in enumerate(zz):
    path2 = os.path.join(path, imagee)
    image2 = ndimage.imread(path2)
    image2=image2.astype(np.float64)
    image2= scipy.misc.imresize(image2, 0.5)
    data[ii,:,:,:]=image2/255

import csv
with open('C:/Users/Hamidreza/Desktop/data/labfall.csv', 'r') as mf:
#with open('C:/Users/Hamidreza/Desktop/FALL DETECTION PROJECT/200 Hz/video/Images/Final/lab.csv', 'r') as mf:
     re = csv.reader(mf,delimiter=',',quotechar='|')
     re=np.array(list(re))
     label = re.astype(np.float64)
     label=np.squeeze(label) 
     
label=np.repeat(label,10)
    
K.set_image_data_format('channels_last')

def CapsNet(input_shape, n_class, routings):

    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='valid', activation='relu', name='conv1')(x)
  #64
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=8, kernel_size=3, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(64, activation='relu', input_dim=8*n_class))#512
    decoder.add(layers.Dense(128, activation='relu'))#1024
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    return train_model, eval_model


def margin_loss(y_true, y_pred):

    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model,x_train,y_train,x_test, y_test, args):
  
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'binary_crossentropy'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'capsnet': 'accuracy'})


    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])

    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, x_test, y_test, args):
    y_pred= model.predict(x_test, batch_size=8)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
            

def load_data():
#    O=2880
#    x_train=data[:O,:,:,:]
#    x_test=data[O:,:,:,:]
#    y_train=label[:O]
#    y_test=label[O:]
#    m=4
#    
#    kf=KFold(5, random_state=None, shuffle=False)
#    kf.get_n_splits(data)
#    k=0
#    for train_index, test_index in kf.split(data):
#        x_train, x_test = data[train_index], data[test_index]
#        y_train, y_test = label[train_index], label[test_index]
#        
#        if k==m:
#           break 
#        k=k+1
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.95)
    y_test2=y_test
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    x_train = x_train.reshape(-1, 64, 68, 3).astype('float32') 
    x_test = x_test.reshape(-1, 64, 68, 3).astype('float32') 
#    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
#    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return x_train, y_train, x_test, y_test, y_test2

    
if __name__ == "__main__":
    import os
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="CapsFall")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=1, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    x_train, y_train, x_test, y_test, y_test2 = load_data()

    input_shape= x_train.shape[1:4] 

    # define model
    model, eval_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, x_test=x_test, y_test=y_test, args=args)
    model=eval_model
    y_pred= model.predict(x_test, batch_size=40)
    y_pred=y_pred[0]
    
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    y_pred = np.argmax(y_pred, axis=1)
    y_pred =y_pred.astype(np.float64)
    labels = {1:'Non-Fall', 2:'Fall'}
    print(classification_report(y_pred, y_test2,
                                target_names=[l for l in labels.values()]))
    r22= metrics.r2_score(y_test2, y_pred)
    print('R2:', r22)