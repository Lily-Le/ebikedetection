# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import cv2
import os
import torch
import matplotlib.pyplot as plt
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import numpy as np
import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2,
                            non_max_suppression, print_args, scale_coords, )
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from graphs import bbox_rel,draw_boxes,draw_boxes_test
import glob

MAX_DET = 1000
HIDE_LABELS = False # hide labels in image
HIDE_CONF = False # hide confidence in view image

class EbikeTracking():
    def __init__(self,weights='runs/train/exp8/best.pt',imgsz=(640, 640),classes=[1,2],config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml",device='cuda:0',conf_thres = 0.25, iou_thres = 0.45, view_img = False):
        ### The class ind for Electric-bike is 1, and the class ind for bicycle is 2, the class ind for person is 0. By default we detect Electric-bike and bicycle
        '''
        weights: 预训练模型路径，为字符串输入
        classes: 检测类别，为列表。默认检测[1,2]为电动车与自行车。
        device: 所使用的GPU编号。
        conf_thresh: 检测的置信度。可以控制只保留置信度高于阈值的检测结果，过滤掉置信度较低的结果。通过调整conf_thresh的值，可以灵活地控制检测结果的准确性和召回率。较高的阈值会过滤掉置信度较低的检测结果，可以提高准确性但可能会降低召回率；较低的阈值会保留更多的检测结果，可以提高召回率但可能会降低准确性。
        view_img: 是否实时显示图像，调试时使用，默认为False。
        config_deepsort: 追踪模型配置文件路径
        iou_thres： 用于设置在目标跟踪中的IoU（Intersection over Union）阈值。IoU是一种衡量两个边界框重叠程度的度量指标。
        在目标跟踪中，当两个边界框的IoU大于等于设定的阈值时，认为这两个边界框表示同一个目标。
        具体来说，当进行目标跟踪时，系统会根据当前帧中的检测结果和前一帧中已经跟踪的目标，通过计算当前帧中每个检测结果与已跟踪目标的IoU，来判断是否将其归为同一个目标。
        如果当前帧中某个检测结果与已跟踪目标的IoU大于等于设定的阈值（iou_thresh），则将其归为同一个目标，否则将其视为新的目标。其余同追踪模型参数。
        '''
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)

        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
        
        self.device = select_device(device)
        self.load_model(weights, imgsz = imgsz)
        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.view_img = view_img # used for debug，view the final results
        # run if tracking from a new stream
        self.tracking_reset()
    
    
    def load_model(self,weights,imgsz=(640, 640)):
        try:
            del self.model
        except:
            print('No existing model to del from GPU')
            
        self.tracking_reset()
        self.model = DetectMultiBackend(weights, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt 
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt else 1, 3, *self.imgsz))
        torch.cuda.empty_cache()


    def tracking_reset(self):
        self.dt, self.seen = [0.0, 0.0, 0.0], 0
        self.frame_idx=0
        self.past_identities=[]

        
    @torch.no_grad()
    def track_one_frame(self, im_rgb):
        # 可以改变这些参数
        max_det = MAX_DET # maximum det 
        hide_labels = HIDE_LABELS
        hide_conf = HIDE_CONF
        # view_img = True
        t1 = time_sync()
        im0 = im_rgb #original sized img
        # resize to required size for the detection model
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = self.model(im, augment=False, visualize=False)
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=max_det)
        self.dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        self.frame_idx=self.frame_idx+1

        DETECT_NEW = False # 是否检测到了新的电动车
        for i, det in enumerate(pred):  # detections per image
            self.seen += 1
 
            s = '%gx%g ' % im.shape[2:]  # print string          
            #check detected boxes, process them for deep sort
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                ##variables for boundig boxes
                bbox_xywh = []
                confs = []
                ## Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                
                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, im0)
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    
                    for i_ in range(outputs.shape[0]):
                        if identities[i_] not in self.past_identities:
                            self.past_identities.append(identities[i_])
                            print(f' Detect E-bike {identities[i_]}')
                            DETECT_NEW = True
             
                    im0=draw_boxes(im0, bbox_xyxy, identities)   #call function to draw seperate object identity
                    # im0=draw_boxes_test(im0, bbox_xyxy, identities = identities, true_pos_str= pos_to_print) 
                annotator = Annotator(im0, line_width=3, example=str(self.names))    
                #yolo write detection result 
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if self.view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (self.names[c] if hide_conf else f'{self.names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            else:
                self.deepsort.increment_ages()
            # Stream results
            if self.view_img:
                cv2.imshow('detect results', im0)
                cv2.waitKey(1)  # 1 millisecond    
        if not DETECT_NEW :
            return None
        else:
            return im0
        
        

    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp8/weights/best.pt', help='model path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=(640,640), help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results') # True if used for debug
    parser.add_argument('--classes', nargs='+', type=int,default = 1,  help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
  
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt
    

def test_rtsp(rtsp_url,save_path,track_interval = 1):
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tracking_ebike = EbikeTracking(**vars(opt))

    frame_count = 0 
    cap = cv2.VideoCapture(rtsp_url)
    # 遍历每一帧
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_count % track_interval == 0:
                result = tracking_ebike.track_one_frame(im_rgb = frame)
                frame_count += 1
                if result is not None:
                    cv2.imwrite(os.path.join(save_path,f'detect_{frame_count}.png'),result)
            else:
                frame_count += 1
        else:
            break
    
    
def test_folder(folder_path, save_path,track_interval = 1):
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tracking_ebike = EbikeTracking(**vars(opt))
    
    frame_count = 0
    imgs =  os.listdir(folder_path)
    imgs.sort()
    for img in imgs:
        if frame_count % track_interval == 0:
            img_path = os.path.join(folder_path, img)
            frame = cv2.imread(img_path)
            result = tracking_ebike.track_one_frame(im_rgb = frame)
            if result is not None:
                cv2.imwrite(os.path.join(save_path,f'detect_{frame_count}.png'),result)
        frame_count += 1

def test_img(img_path, save_path):
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    tracking_ebike = EbikeTracking(**vars(opt))
    
    frame = cv2.imread(img_path)
    result = tracking_ebike.track_one_frame(im_rgb = frame)
    if result is not None:
        cv2.imwrite(os.path.join(save_path,f'detect.png'),result)

if __name__ == "__main__":
    test_folder('test_samples','runs/exp_tracking')

    
