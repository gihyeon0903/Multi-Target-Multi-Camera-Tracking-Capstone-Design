## Sort Usage

### Dependencies
- Refer to <a href="https://github.com/ultralytics/yolov5">github1</a>
- Refer to <a href="[https://github.com/ultralytics/yolov5](https://github.com/abewley/sort)">github2</a>

### Code

#### 1. Sort
~~~
from sort import *
mot_tracker = Sort(max_age, min_hits, iou_threshold)
~~~
> max_age  : Maximum number of frames to keep alive a track without associated detections.<br>
> min_hits : Minimum number of associated detections before track is initialised.<br>
> iou_threshold : Minimun IOU for match
  
#### 2. yolov5s 
~~~
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True);
model.float()
model.eval();
~~~

#### 3. tracking
~~~
preds = model(image_show)
detections = preds.pred[0].to('cpu').numpy()
track_bbs_ids = mot_tracker.update(detections)
~~~
1. yolov5를 이용한 Object Detection
  > detections    : [x1, y1, x2, y2, confidence, class]
2. Sort(mot_tracker)를 이용한 Object Detection
  > track_bbs_ids : [x1, y1, x2, y2, Id]
   


