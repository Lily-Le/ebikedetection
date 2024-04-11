import cv2
import numpy as np
import torch
import argparse
from utils.general import print_args
import os
import importlib
import sys
from pathlib import Path
from models.common import DetectMultiBackend

from utils.plots import Annotator, colors
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode

class EbikeDetection(): 
    def __init__(self,model_path = 'runs/train/exp8/weights/best.pt',classes = [1,2],device=0,conf_thresh = 0.25,view_img = False):
        '''
        params: 
            model_path: 预训练模型路径，为字符串输入
            classes: 检测类别，为列表。默认检测[1,2]为电动车与自行车。
            device: 所使用的GPU编号。
            conf_thresh: 检测的置信度。可以控制只保留置信度高于阈值的检测结果，过滤掉置信度较低的结果。通过调整conf_thresh的值，可以灵活地控制检测结果的准确性和召回率。较高的阈值会过滤掉置信度较低的检测结果，可以提高准确性但可能会降低召回率；较低的阈值会保留更多的检测结果，可以提高召回率但可能会降低准确性。
            view_img: 是否实时显示图像，调试时使用，默认为False。
        '''
        try:
            spec = importlib.util.find_spec('yolov5')
        except AttributeError:
            print(
                "package 'yolov5' not found, you could install it with pip",
                file=sys.stderr,
            )
            sys.exit(1)
        else:
            self.yolo_pkgdir = Path(spec.origin).parent.resolve()
        
        self.model = torch.hub.load(
            str(self.yolo_pkgdir),
            "custom",
            path=model_path,
            source="local",
            device=device,
        )
        self.device = device
        self.classes = classes
        self.conf_thresh = conf_thresh
        self.view_img = view_img

    def detect_one_frame(self, frame):
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        w, h = frame.shape[1], frame.shape[0]
        n = len(labels) 
        DETECT = False
        for i in range(n):
            row = cord[i]
            name = self.model.names[int(labels[i])]
            if (int(labels[i]) in self.classes) and row[4].item() >= self.conf_thresh:
                DETECT = True
                x1, y1, x2, y2 = (
                    int(row[0] * w),
                    int(row[1] * h),
                    int(row[2] * w),
                    int(row[3] * h),
                )
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 3)
                cv2.putText(
                    frame,
                    name,
                    (x1+10, y1+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bgr,
                    2,
                )
        
        if self.view_img:
            cv2.imshow('detect results', frame)
            cv2.waitKey(1)  # 1 millisecond    
        if DETECT:
            return frame
        else:
            return None




def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='runs/train/exp8/weights/best.pt', help='path to model')
    parser.add_argument('--classes', type=int, nargs='+', default=[1,2], help='classes to detect. Here by default 1: Electric-bicycle; 3: bicycle; 4:person')
    parser.add_argument('--device', type=int, default=0, help='device index')
    parser.add_argument('--conf-thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--view-img', action='store_true', help='display detection results or not')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def test_rtsp(rtsp_url,save_path,frame_interval = 1):
    # test video: simply replace rtsp_url with the path of your video
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    detecting_ebike = EbikeDetection(**vars(opt))
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_interval == 0:
                frame = detecting_ebike.detect_one_frame(frame)
                if frame is not None:
                    cv2.imwrite(os.path.join(save_path,f'detect_{frame_count}.png'),frame)
            frame_count += 1
        else:
            break

def test_folder(folder_path, save_path,frame_interval =1):
    
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    detecting_ebike = EbikeDetection(**vars(opt))
    frame_count = 0
    for img in os.listdir(folder_path):
        if frame_count % frame_interval == 0:
            img_path = os.path.join(folder_path, img)
            frame = cv2.imread(img_path)
            frame = detecting_ebike.detect_one_frame(frame)
            if frame is not None:
                cv2.imwrite(os.path.join(save_path,f'detect_{frame_count}.png'),frame)
        frame_count += 1

def test_img(img_path, save_path):
    opt = parse_opt()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    detecting_ebike = EbikeDetection(**vars(opt))
    frame = cv2.imread(img_path)
    frame = detecting_ebike.detect_one_frame(frame)
    if frame is not None:
        cv2.imwrite(os.path.join(save_path,f'detect.png'),frame)

if __name__ == "__main__":
    test_folder('test_samples','runs/exp_detection')
    # test_img('dataset/CV_ebike/dianche3080/images/2020(17).jpg','runs/exp_detection')


