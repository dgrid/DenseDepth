import os
import sys
import glob
import argparse
import matplotlib
import cv2
import time
from matplotlib import pyplot as plt

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images, predict_test
from loss import depth_loss_function

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
parser.add_argument('--weight', type=str, help='weight file[.hdf5]')
parser.add_argument('--input_list', default='None', type=str, help='samples[.csv]')
parser.add_argument('--batch_size', default=4, type=int)
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# set weight 2020/02/12
if args.weight != None:
    print("get weights from %s" % args.weight)
    model.load_weights(args.weight, by_name=False)

# Input images
if args.input_list == 'None':
    names = glob.glob(args.input)
    inputs = load_images( names )
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

    for i in range(len(outputs)):
        basename = os.path.basename(names[i])
        path = names[i][:len(names[i])-len(basename)]
        time.sleep(0.1)
        cv2.imwrite(os.path.join(path, 'inferred_' + basename[:-3] + 'pfm'), outputs[i])
    #matplotlib problem on ubuntu terminal fix
    #matplotlib.use('TkAgg')

    # Display results
    viz = display_images(outputs.copy(), inputs.copy())
    plt.figure(figsize=(10,5))
    plt.imshow(viz)
    plt.savefig('test.png')
    #plt.show()
    sys.exit(0)

with open(args.input_list, 'r') as f:
    data = f.readlines()

data_count = len(data)
used_data = 0
while data_count > used_data:
    names = []
    for i in range(min(args.batch_size, data_count-used_data)):
        names.append(data[used_data + i].strip())
    inputs = load_images(names)
    print('\nLoarded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
    outputs = predict(model, inputs)
    for i in range(len(outputs)):
        basename = os.path.basename(names[i])
        path = names[i][:len(names[i])-len(basename)]
        time.sleep(0.1)
        cv2.imwrite(os.path.join(path, 'inferred_' + basename[:-3] + 'pfm'), outputs[i])
    used_data += args.batch_size
    
