import cv2
import numpy as np
from openvino.inference_engine import IECore
import time

# Set paramters
model_xml = 'emotions-recognition-retail-0003.xml'
model_bin = 'emotions-recognition-retail-0003.bin'
device = 'CPU' #'MYRIAD'  #device to run the inference on
image_interval = 3  #seconds between frame catpure
image_max = 10      #number of images to capture from video
count = 1
emotion = {0:'neutral', 1:'happy', 2: 'sad', 3:'surprise', 4:'anger'}

# Load the model.
ie = IECore()
net = ie.read_network(model=model_xml, weights=model_bin)

# Loading model to the plugin 
exec_net = ie.load_network(network=net, device_name=device)

# Gather layer names from net.inputs and net.outputs
# The net.inputs object is a dictionary that maps input layer names to DataPtr objects.
input_blob = next(iter(net.inputs))  #'data'
out_blob = next(iter(net.outputs))   #'prob_emotion'

# Read network input shape
n, c, h, w = net.inputs[input_blob].shape  

# Open the device at the ID 0
cap = cv2.VideoCapture(0)

#Check whether user selected camera is opened successfully.
if not (cap.isOpened()):
    print('Could not open video device')

#To set the resolution    
cap.set(3, 640)
cap.set(4, 480)


while(True):
    # Capture frame-by-frame
    while count <= image_max:
  
        print ('Process new frame: ', count)
        ret, frame = cap.read()
        
        #pre-process images for model input
        images = np.ndarray(shape=(n, c, h, w))
        image = frame

        if image.shape[:-1] != (h, w):
            print('Image resized from {} to {}'.format(image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        
        # Change data layout from HWC to CHW
        images = image.transpose((2, 0, 1))  
    
        # Start inference
        res = exec_net.infer(inputs={input_blob: images})
        probs = np.squeeze(res[out_blob])
        emotion_cap= emotion[np.argmax(probs, axis=-1)]
        
        #Print inference probabilities
        print ('        neutral    happy      sad        surprise   anger')
        print ('probs:', probs)
        print ('emotion: ', emotion_cap)
        cv2.imwrite('Frame_{}_{}.jpg'.format(count, emotion_cap), frame)
        count += 1
        time.sleep(image_interval)
           
    if cv2.waitKey(0): #break out of loop (rework to improve)
        break
        
cv2.VideoCapture(0).release()

