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
<p align="center">
  <img src="./results/4 corner detection.png" width="700" height="90"/>
</p>
Additionally, We add k to similarity matrix. It can include IOU information to ID matching. <br>
<p align="center">
  <img src="./results/4 corner detection.png" width="700" height="90"/>
</p>
The equation for k is as follows. <br>
$ \Alpha \rightarrow \Omega $

--------------
### Result
<p align="center">
<img src="./results/result.png" width="630" height="350"/>
</p>

