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

#### 3. Classification
We classify car plate type using custom model. (1996_b, 2004_n, ....)<br>
<p align="center">
  <img src="./results/classification.png" width="500" height="120"/>
</p>

#### 4. De-Identification
We make natural car plate using De-Identification technique in three steps below.<br>
- 1. Generate artificial plate
- 2. Transfer style of original plate to artificial plate
- 3. Synthesize plate

<br>

> #### i. Generate artificial plate
> - Define Position of {Digit(숫자), Word(문자), Region(지역)} in plate.<br>
> - Randomly generates characters that may appear depending on the type of license plate.<br>
> - Sampling image matching Generated characters from DataBase.<br>
> - Attach image to car plat frame
<p align="center">
  <img src="./results/de identification1.png" width="300" height="110"/>
  <img src="./results/de identification2.png" width="300" height="80"/>
  <img src="./results/de identification3.png" width="270" height="150"/>
</p>
<br>

> #### ii. Transfer style of original plate to artificial plate
> - We use 3-models for style transfer (Style Transfer, Pix2Pix, Style Swap)
> - We evaluate the 3-models using PSNR and SSIM metrics.
<p align="center">
<img src="./results/transfer1.png" width="630" height="270"/>
</p>
<br>

> #### iii. Synthesize plate(excepted Pix2Pix model)
> - We synthesize original image and generated car plate image using inverse perpective transfrom and seamless clone filter
<p align="center">
<img src="./results/synthetic1.png" width="630" height="270"/>
</p>
<br>

--------------
### Result
<p align="center">
<img src="./results/result.png" width="630" height="350"/>
</p>

