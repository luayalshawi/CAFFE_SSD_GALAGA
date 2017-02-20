
import numpy as np
import cv2
import time
import pyscreenshot as ImageGrab
from subprocess import Popen, PIPE
from pykeyboard import PyKeyboard
from random import randint
import unirest
cap = cv2.VideoCapture(0)
k = PyKeyboard()
import json
import requests
url = 'http://192.168.0.103:5000/api/sendkeys/'
headers = {'content-type': 'application/json'}
# def keypress(cmd):
#     k.tap_key(cmd)
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



model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_12000.caffemodel'

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
    dict = {}
    # img = ImageGrab.grab(bbox=(0,0,1680,1050),backend='scrot') #bbox specifies specific region (bbox= x,y,width,height)
    # frame = np.array(img)
    flag, frame = cap.read()
    start = time.time()
    if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        # If the number of captured frames is equal to the total number of frames,
        # we stop
        break
    numerate = 0

    image = frame#caffe.io.load_image('examples/images/gal1.jpg')




    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    # Forward pass.
    detections = net.forward()['detection_out']

    # Parse the outputs.
    det_label = detections[0,0,:,1]
    det_conf = detections[0,0,:,2]
    det_xmin = detections[0,0,:,3]
    det_ymin = detections[0,0,:,4]
    det_xmax = detections[0,0,:,5]
    det_ymax = detections[0,0,:,6]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.20]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_labels = get_labelname(labelmap, top_label_indices)
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    print "image.shape:%d" %(image.shape[1])
    print "top_conf.shape:%d" %(top_conf.shape[0])
    for i in xrange(top_conf.shape[0]):
        xmin = long(round(top_xmin[i] * image.shape[1]))
        ymin = long(round(top_ymin[i] * image.shape[0]))
        xmax = long(round(top_xmax[i] * image.shape[1]))
        ymax = long(round(top_ymax[i] * image.shape[0]))
        score = top_conf[i]
        label = long(top_label_indices[i])
        label_name = top_labels[i]
        display_txt = '%s: %.2f'%(label_name, score)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        print "coords: %d %d %d %d" %(xmin, ymin, xmax-xmin+1, ymax-ymin+1)
        if label_name == "player" and xmin <= 1069:
            dict[label_name] =  xmin, ymin
        elif label_name != "player" :
            dict[label_name+str(i)] =  xmin, ymin
        #cv2.rectangle(frame,(xmin,ymin),(xmax, ymax),(0,255,0),1)

        #cv2.putText(frame,display_txt, (xmin,ymin), cv2.FONT_HERSHEY_SIMPLEX, 1,200)

        # cv2.rectangle(frame,(1680/2,1050/2),(10, 10),(0,255,0),1)

        cv2.imshow('video', frame)

        cv2.waitKey(1)

    # decision
    if "player" in dict.keys():
        print "decision"
        px,py = dict["player"]
        for key,(x,y) in dict.iteritems():
            print key,x,y
            if key != "player":
                del_x = px - x
                y_threshold = 150
                threshold = 50
                if y >= y_threshold:
                    if px - del_x >0 and px - del_x <= threshold:
                        print "Key press Right"
                        payload = {'keys': '{Right}'}
                        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
                        print response
                        break
                    if px - del_x < 0 and px - del_x <= -1*threshold:
                        print "Key press left"
                        payload = {'keys': '{Left}'}
                        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
                        print response
                        break
                    if px - del_x == 0:
                        print 'shoot - enemy close to player on y coordinate'
                        payload = {'keys': "{f}"}
                        response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
                        print response
                        break
                if px - del_x >= -1*threshold or px - del_x <= threshold:
                    print 'shoot'
                    payload = {'keys': "{f}"}
                    response = unirest.post(url, params=json.dumps(payload), headers=headers, callback=callback_function)
                    print response
                    break
    cv2.imshow('video', frame)

    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()
