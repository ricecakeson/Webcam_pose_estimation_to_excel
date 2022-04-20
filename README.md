# Webcam_pose_estimation_to_excel
from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2
from pyparsing import delimited_list, delimitedList

import numpy as np
import mxnet as mx
from mxnet import gluon, image, ndarray_doc
from mxnet import ndarray as nd
from mxnet.gluon.data.vision import transforms
import pandas as pd
#import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints
import matplotlib.pyplot as plt

ctx = mx.cpu()
detector_name = "yolo3_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimators = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimators.hybridize()

cap = cv2.VideoCapture(0)
time.sleep(1)  ### letting the camera autofocus

axes = None
num_frames = 100

#pred_coords_2d=np.zeros((0,34))
pred_coords_2d =np.empty((1,34), float)
sample_num = 0
captured_num = 0
for i in range(num_frames):
    ret, frame = cap.read()
    sample_num = sample_num + 1
    if not ret:
        break

    if sample_num == 1:
        captured_num = captured_num + 1
        cv2.imwrite('./images/img'+str(captured_num)+'.jpg',frame)
        sample_num = 0
    inversed = ~frame
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx) 

    class_IDs, scores, bounding_boxs = detector(x)
    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimators(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        
        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.2)
       
        print('     3150 pred_coords type=',type(pred_coords))
        print('     3152 pred_coords ndim=',pred_coords.ndim)
        print('     3154 pred_coords shape=',pred_coords.shape)
        #pred_coords_trans = nd.array([-1])
        #for _ in range(31):
        #pred_coords_trans = pred_coords.expand_dims(axis=0)
        #pred_coords_trans1 = pred_coords_trans
        print(pred_coords)
        print('     3170 pred_coords_2d_1 shape=',type(pred_coords))

        pred_coords_2d_1 = pred_coords.reshape(pred_coords.shape[0], -1)   
        np_ex_float_array = pred_coords_2d_1.asnumpy()
        
        print('     3170 pred_coords_2d_1 shape=',np_ex_float_array.shape)
        pred_coords_2d = np.append(pred_coords_2d , np_ex_float_array, axis = 0)
        #print('     3190 pred_coords_2d_1 shape=',pred_coords_2d_1.shape)
        #print('     3192 pred_coords_2d shape=',pred_coords_2d.shape)
        #pred_coords_2d = np.append(pred_coords_2d,pred_coords_2d_1, axis=0)
        
        #print('     3190 pred_coords_2d shape=',type(pred_coords_2d))
        #pred_coords_3d = pred_coords.reshape(pred_coords.shape[2], -1)
        
        #print('     3154 pred_coords_2d shape=',pred_coords_2d.shape)
        
        #print(pred_coords_2d[:1,:1])
        #print(pred_coords_2d)
        #print(pred_coords_3d)
        

        #df = pd.DataFrame(list(pred_coords_2d[:1,:1]))
        #df.to_csv("test2.csv",header='col',index=None)

        #np.savetxt('TopCoatMain_bb.csv', list(pred_coords_3d), delimiter=',')
        df = pd.DataFrame(pred_coords_2d)
        df.columns = ['nose_x', 'nose_y','left_eye_x','left_eye_y','right_eye_x', 'right_eye_y','left_ear_x', 'left_ear_y','right_ear_x', 'right_ear_y','left_shoulder_x', 'left_shoulder_y','right_shoulder_x', 'right_shoulder_y','left_elbow_x', 'left_elbow_y','right_elbow_x', 'right_elbow_y','left_wrist_x', 'left_wrist_y','right_wrist_x', 'right_wrist_y','left_hip_x', 'left_hip_y','right_hip_x', 'right_hip_y','left_knee_x', 'left_knee_y','right_knee_x', 'right_knee_y','left_ankle_x', 'left_ankle_y','right_ankle_x', 'right_ankle_y']
        df.to_excel('data.xlsx')
        csv_data = pd.read_excel('data.xlsx')

        
        #fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        #delay = round(1000/fps)
        #out = cv2.VideoWriter('SaveVideo.mp4',fourcc,fps,(640, 480))

     
    
    
        
        
    cv_plot_image(img)
    

    cv2.waitKey(1)

    
    
cap.release()
