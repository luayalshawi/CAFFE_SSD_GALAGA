import numpy as np
import cv2
import time
import unirest
import cv2.cv as cv
import pyscreenshot as ImageGrab

import json
import requests
url = 'http://192.168.0.100:5000/api/sendkeys/'
headers = {'content-type': 'application/json'}
def callback_function(response):
  response.code # The HTTP status code
  response.headers # The HTTP headers
  response.body # The parsed response
  response.raw_body # The unparsed response
# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
#
# os.environ["GLOG_minloglevel"] ="3"
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames



model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy_decision.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_3000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

framdeid = 0

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
# transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB



# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,3,image_resize,image_resize)
# plt.show()
while True:

    img = ImageGrab.grab(bbox=(500,120,1040,800),backend='scrot') #bbox specifies specific region (bbox= x,y,width,height)
    frame = np.array(img)
    start = time.time()

    #img.save('screenshot.png')
    numerate = 0

    image = frame
    #image = adjust_gamma(frame)




    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    decision = net.forward()['decision']

    if(decision[0]==0.0):
        print "right"
        payload = {'keys': 'x'}
        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
        print response
    elif decision[0]==1.0:
        print "left"
        payload = {'keys': 'z'}
        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
        print response
    elif decision[0]==2.0:
        print "shoot"
        payload = {'keys': "f"}
        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
        print response
    elif decision[0]==10.0:
        print "right _x"
        payload = {'keys': "_x"}
        response = unirest.post(url, params=json.dumps(payload), headers=headers)#, callback=callback_function)
        print response
    elif decision[0]==100.0:
        print "left _z"
        payload = {'keys': "_z"}
        response = unirest.post(url, params=json.dumps(payload), headers=headers)#, callback=callback_function)
        print response
    cv2.imshow('video', frame)

    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
