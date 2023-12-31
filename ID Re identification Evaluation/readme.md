# Detection, Traker, Re-id matching using Answer(json)
- Detection (x1, y1, x2, y2, Conf_score, class)
- Traker (x1, y1, x2, y2, local_id)
- Re-id matching (matched, unmatched)

### Dataset
<p align="center">
  <img src="./result/result2.PNG" width="350"/>
</p>
다운로드 링크 : https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/

### Usage
#### 1. Detection & Trackers
~~~
## Trackers
trks1 = convert_trks_shape(objs1)
trks2 = convert_trks_shape(objs2)
~~~
json -> (x1, y1, x2, y2, Conf_score, class) -> (x1, y1, x2, y2, local_id) <br>
정답 데이터를 yolo의 출력 shape으로 변경 후 sort의 출력 shape로 변환

#### 2. Find Similarity Matrix
~~~
## find similarity matrix
similarity_matrix = calculate_similarity_matrix(trks1, trks2)
~~~
정답 데이터의 global id를 이용하여 cam1-cam2에서 matching 되는 객체(row, col)의 요소에 0.9 나머지에는 0으로 초기화함 (가상의 simliarity_matrix 생성)

#### 3. Re-id matching (with Hungarian algorithm)
~~~
## find match, unmatch use Hungarian algorithm
assignments, cos_similarities = calculate_hungarian_algorithm(similarity_matrix)
matched   = assignments[cos_similarities > 0.5]
unmatched = calculate_non_matched(trks1, trks2, matched)
~~~
헝가리안 알고리즘을 통해, 최적의 matching을 구하고 그 때의 유사도 결과가 0.5 이상인 쌍만을 matched에 넣어줌.<br>
matched되지 않은 나머지는 unmatched에 넣어줌.

### Result

<p align="center">
  <img src="./result/result1.PNG" width="350"/>
</p>

- 2번째 프레임에서의 Re-id matching 결과는 아래와 같음<br>
  - cam1에는 0, 1, 2, 3 총 4개의 객체가 존재하고 cam2에는 0, 1 총 2개의 객체가 존재
  - cam1과 cam2에서 0과 0은 동일한 객체
  - cam1에서 1, 2, 3은 cam1에만 존재하는 객체
  - cam2에서 1은 cam2에만 존재하는 객체
  - 여기서 [0, 1, 2, 3]과 [0, 1]은 local id의 인덱스를 의미하며 이를 이용하여 local id에 접근할 수 있음.
