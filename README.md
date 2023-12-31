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
cd osnet/
cp osnet_x1_0.onnx osnet/osnet_x1_0.onnx

cd feature_extractor_weights/
cp osnet_ain_ms_m_c.pth.tar feature_extractor_weights/osnet_ain_ms_m_c.pth.tar

~~~
<a href="https://drive.google.com/drive/folders/1XhUQNov126Qfns1RgYb5HE8F6wzxX-wu?usp=drive_link"</a>
  
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
|---|:---:|:---:|:---:|
|**ID 11**|0.6|0.7|0|0|
|**ID 12**|0.7|0.6|0|0|

> cam1 target(ID 1,2,3,4), cam2 target(ID 11,12)
> Object1 (ID 1 - ID 11), Object2 (ID 2 - ID 12)
<br>

Additionally, We add k to similarity matrix. It can include IOU information to ID matching. <br>

|ID 1|ID 2|ID 3|ID 4|
|---|:---:|:---:|:---:|
|**ID 11**|0.6+k|0.7|0|0|
|**ID 12**|0.7|0.6+k|0|0|

> cam1 target(ID 1,2,3,4), cam2 target(ID 11,12)
> Object1 (ID 1 - ID 11), Object2 (ID 2 - ID 12)
<br>

The equation for k is as follows. <br>

<p align="center">
  <img src="./images/equation.png" width="300"/>
</p>

#### 3. Integrate Local IDs(cam1, cam2) to Global IDs
Third, We propose a way to manage global IDs, which consists of a total of three Action.
> 1. Generate <br>
> Generate new global ID.
> 2. Update <br>
> Modify local ID maintaining global ID.
> 3. Delete <br>
> Delete existing global ID.
<br>

<p align="center">
  <img src="./images/action1.png" width="900"/>
</p>

--------------
### Result

<p align="center">
  <img src="./images/result.gif" width="300"/>
</p>
