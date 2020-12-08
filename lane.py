# 모듈 호출
import numpy as np
import cv2 as cv
import os
import imutils
import argparse
import os.path as ops
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import random

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

prevTime = 0
weights_path= "./weight2/tusimple_lanenet.ckpt"
CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

########### YOLO 관련 ############
# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image 더 빠른 결과 320, 더 정확한 결과 608

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f : classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

def imshow(tit, image) :
    plt.title(tit)    
    if len(image.shape) == 3 :
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    else :
        plt.imshow(image, cmap="gray")
    plt.show()
        
def create_win(frames, scale=1.0) :    
    global myImage
    
    all = []
    for f in frames :
        if len(f.shape ) !=  3 : f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
        all.append(f)
    frame = np.vstack(all)
    
    fr=cv.cvtColor(frame, cv.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
    fr=cv.flip(fr, 0) # because Bokeh flips vertically
    width=fr.shape[1]
    height=fr.shape[0]    

    p = figure(x_range=(0,width), y_range=(0,height), output_backend="webgl",
               width=int(width*scale), height=int(height*scale))    
    myImage = p.image_rgba(image=[fr], x=0, y=0, dw=width, dh=height)
    show(p, notebook_handle=True)   
    
def update_win(frames) :    
    all = []
    for f in frames :
        if len(f.shape ) !=  3 : f = cv.cvtColor(f, cv.COLOR_GRAY2BGR)
        all.append(f)
    frame = np.vstack(all)
    
    fr=cv.cvtColor(frame, cv.COLOR_BGR2RGBA)
    fr=cv.flip(fr, 0)
    myImage.data_source.data['image']=[fr]
    push_notebook()

############################################################################

def minmax_scale(input_arr):
    """
    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr



# VideoCapture 객체 정의
cap = cv.VideoCapture('test4.mp4')
# 동영상
fourcc = cv.VideoWriter_fourcc(*'DIVX')
# 프레임 너비/높이, 초당 프레임 수 확인
width = cap.get(cv.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv.CAP_PROP_FPS) # 또는 cap.get(5)
#print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))
out = cv.VideoWriter('out4.mp4', fourcc, fps, (int(width), int(height))) # VideoWriter 객체 정의

input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
net = lanenet.LaneNet(phase='test', cfg=CFG)

binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet') ## value error

postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

sess_config = tf.ConfigProto()
sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
sess_config.gpu_options.allocator_type = 'BFC'
sess = tf.Session(config=sess_config)


with tf.variable_scope(name_or_scope='moving_avg'):
    variable_averages = tf.train.ExponentialMovingAverage(
        CFG.SOLVER.MOVING_AVE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

with sess.as_default():
    saver.restore(sess=sess, save_path=weights_path)
    while cap.isOpened(): # cap 정상동작 확인
        ret, frame = cap.read()
        # 프레임이 올바르게 읽히면 ret은 True
        if not ret:
            print(ret)
            print("프레임을 수신할 수 없습니다(스트림 끝?). 종료 중 ...")
            break
        frame = imutils.resize(frame, width=1280,height=720)

        image_vis = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # image_vis = frame




        image = cv.resize(frame, (512,256), interpolation=cv.INTER_LINEAR)
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = image / 127.5 - 1.0

        binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [image]} )


        postprocess_result = postprocessor.postprocess(
            min_area_threshold = 150,
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis)
        mask_image = postprocess_result['mask_image']
        
        #print(CFG.MODEL.EMBEDDING_FEATS_DIMS) # 4

        # for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
        #     instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        # embedding_image = np.array(instance_seg_image[0], np.uint8)
#         mask_image = postprocess_result['mask_image']
#         source_image = postprocess_result['source_image']
#         print(frame.shape) #(256, 512, 3)
#         print(embedding_image.shape) # (256, 512, 4)
#         print(image_vis[:, :, (2, 1, 0)].shape) #(277, 512, 3)
#         print(embedding_image[:, :, (2, 1, 0)].shape) # (256, 512, 3)
#        print(mask_image[:, :, (2, 1, 0)].shape) #(256, 512, 3)
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        
        speed =  "FPS : %0.1f" % fps
        image_vis = cv.resize(image_vis,(512,256))
        # tt=cv.addWeighted(mask_image[:, :, (2, 1, 0)],0.6,embedding_image[:, :, (2, 1, 0)],0.4,0)
        try:
            tt=cv.addWeighted(mask_image[:, :, (2, 1, 0)],0.3,image_vis[:, :, (2, 1, 0)],0.7,0)

            # cv.imshow('Otter', mask_image[:, :, (2, 1, 0)])
        except Exception as e:
            pass
        
        cv.putText(tt,speed, (0, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)

        blob = cv2.dnn.blobFromImage(tt, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False) # yolo 부분 시작

        #for (x, y, w, h) in blob : # full-body만 검출 가능
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        postprocess(tt, outs)

        out.write(tt)
        cv.imshow('Otter', tt) 


        #cv.imshow('Otter', embedding_image[:, :, (2, 1, 0)])
        #cv.imshow('Otter', image_vis[:, :, (2, 1, 0)])
        #cv.imshow('Otter', mask_image[:, :, (2, 1, 0)])
        
        if cv.waitKey(42) == ord('q'):
            break
sess.close()
# 작업 완료 후 해제
cap.release()
out.release()

cv.destroyAllWindows()