# Multi-Target-Multi-Camera-Tracking-Capstone-Design

### Project Goal
- Multi Target Tracking using 2 cameras
--------------

### Dependencies
- numpy 1.2+
- opencv-python 4.8+
- pytorch 2.0+
--------------
~~~
import os

os.makedirs(data/model_weights)
cd data/model_weights
~~~
<a href="https://drive.google.com/drive/folders/1mbzC1hsGuE-jEdX0ImvXI4tSeNb3Il36?usp=drive_link">download model weights</a>
  
### Method
#### 1. Single Camera Object Tacking
Fisrt, We perform single camera object tracking using (yolov5+sort) about cam1 and cam2. <br>
<p align="center">
  <img src="./results/plate detection.jpg" width="300" height="350"/>
  <img src="./results/plate detection.jpg" width="300" height="350"/>
</p>

#### 2. Re-ID
Second, We extract features using pre-trained weight of OSnet and match same ID to campare features using cosine similarity <br>

|ID 1|ID 2|ID 3|ID 4|
|---|---|---|---|
|**ID 11**|0.6|0.7|0|0|
|**ID 12**|0.7|0.6|0|0|

> cam1 target(ID 1,2,3,4), cam2 target(ID 11,12)
> Object1 (ID 1 - ID 11), Object2 (ID 2 - ID 12)
<br>

Additionally, We add k to similarity matrix. It can include IOU information to ID matching. <br>

|ID 1|ID 2|ID 3|ID 4|
|---|---|---|---|
|**ID 11**|0.6+k|0.7|0|0|
|**ID 12**|0.7|0.6+k|0|0|

> cam1 target(ID 1,2,3,4), cam2 target(ID 11,12)
> Object1 (ID 1 - ID 11), Object2 (ID 2 - ID 12)
<br>

The equation for k is as follows. <br><br>
<p align="center">
  $k=\frac{\min(\max\_hits, hits)\cdot0.2}{\max\_{hits}}$
</p>

#### 3. Integrate Local ID(cam1, cam2) to Global ID



--------------
### Result

<p align="center">
<img src="./results/result.png" width="630" height="350"/>
</p>

