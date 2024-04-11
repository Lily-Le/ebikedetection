# 使用说明

本项目用于识别电梯监控视角内的电动车以及自行车。提供了基于检测的方法与基于跟踪的方法。基于检测的方法会对检测到有目标实例的每一帧返回标注图像， 基于跟踪的方法则在此基础上进行了去重。 

运行环境：
```sh
conda  activate  yolo
```

## 预训练模型

对yolov5s进行了微调，以增加在电梯视角下识别的准确率。

**模型路径**： `runs/train/exp8/weights/best.pt`

**可识别类别**： 电动车(类别号1)，自行车（类别号2），行人（类别号4）

## 基于检测的方法

运行 `detect_ebike.py` 文件,  

`detect_ebike.py`中提供的`EbikeDetection`类，用于检测电动车以及自行车。

```python
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
    def detect_one_frame(self, frame):
        '''
        执行检测的主函数，仅当检测结果不为空时返回标注后的图像，否则返回None。
        '''
```
- **使用方法**  
1. 创建检测实例

    ```python
    detecting_ebike = EbikeDetection(model_path = 'runs/train/exp2/weights/best.pt',classes = [1,2],device=0,conf_thresh = 0.25,view_img = False)
    ```
    可全部使用默认参数 ```detecting_ebike =  EbikeDetection() ```

2. 逐帧识别图像：  

    使用时调用`detecting_ebike.detect_one_frame()`函数。  
    ```python
    result = detecting_ebike.detect_one_frame(frame)
    ```  
    `frame`为从rtsp流/视频文件/图像文件等中截取的一帧图像。  
    当检测结果不为空时，返回标注好的图像，否则返回None。

- **示例**  

    `detect_ebike.py`文件中提供了三个示例函数运行，分别接收 rtsp流/视频文件，文件夹，以及单帧图像作为输出。  
    **rtsp流/视频文件：** 
    ```python
    def test_rtsp(rtsp_url,save_path,frame_interval = 1):
        '''
        rtsp_url: rtsp流地址。
        save_path: 结果文件存储路径，存储所有检测结果不为空的标注图像，用于调试。
        frame_interval: 间隔多少帧进行检测。
        '''
    ```
    **文件夹：**
    ```python
    def test_folder(folder_path, save_path,frame_interval =1)：
        '''
        folder_path: 存储测试图像的文件夹，默认其中的所有文件为图像格式。
        其余同test_rtsp函数。
        '''
    ```
    **单帧图像：**
    ```python
    def test_img(img_path, save_path):
    ```
    **Note**  

    `detect_ebike.py`文件默认执行`test_folder`函数，`test_sample`提供了少许测试图像。

    `parse_opt()`函数用以命令行接收`EbikeDetection`类的初始化参数，用以测试时使用。初始化参数可不作修改。
    
## 基于跟踪的方法

基于跟踪的方法不会对检测到的每一帧返回标注图像，只会在检测到新的实例后返回标注图像。

具体可见
 `track_ebike.py` 文件。定义了`EbikeTracking`类用以跟踪检测对象,使用方法与基于检测的方法类似，可均使用默认参数。
 ```python 
 tracking_ebike = EbikeTracking(weights='runs/train/exp8/weights/best.pt',imgsz=(640, 640),classes=[1,2],config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml",device='cuda:0',conf_thres = 0.25, iou_thres = 0.45, view_img = False)
 # 默认参数： tracking_ebike = EbikeTracking()
 '''
 config_deepsort: 追踪模型配置文件路径
 iou_thres： 
    用于设置在目标跟踪中的IoU（Intersection over Union）阈值。IoU是一种衡量两个边界框重叠程度的度量指标。在目标跟踪中，当两个边界框的IoU大于等于设定的阈值时，认为这两个边界框表示同一个目标。
    具体来说，当进行目标跟踪时，系统会根据当前帧中的检测结果和前一帧中已经跟踪的目标，通过计算当前帧中每个检测结果与已跟踪目标的IoU，来判断是否将其归为同一个目标。如果当前帧中某个检测结果与已跟踪目标的IoU大于等于设定的阈值（iou_thresh），则将其归为同一个目标，否则将其视为新的目标。
其余同检测模型参数。
 '''
 result = tracking_ebike.track_one_frame(frame)
 ```

识别示例结果：

<img src="figs/detect.png" alt="Image" width="400">
