## Sort Usage

### Dependencies
- Refer to <a href="https://github.com/ultralytics/yolov5">github</a>

### Code

#### 1. Sort tracker
~~~
from sort import *
mot_tracker = Sort(max_age=10)
~~~

#### 2. yolov5s 
~~~
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True);
model.float()
model.eval();
~~~

####
