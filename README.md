# Modal-Importance-based-Improved-lifelog-prediction-performance

## 연구 목표 
- 본 연구는 '라이프로그 예측 성능 향상을 위한 다중 인스턴스 기반 개인별 모달 주요도 추출 및 그룹화' 논문에서 구현한 실험을 재현할 수 있게 하는 코드입니다.
- 멀티 모달 센서 데이터를 활용해서 사용자들의 라이프로그를 예측하는 문제는 개인의 다양성으로 인해 발생되는 편차로 인해 어려움이 있습니다.
- 본 연구에서는 각 개인에게서 사용자들의 라이프로그 예측에 영향을 많이 주는 모달 주요도를 설명가능한 모델인 Mulitple Instance learning을 최초로 사용해서 추출하고, 추출한 모달 주요도를 사용해서 그룹별 예측 모델이 전체 데이터를 사용한 방법론보다 성능이 좋다는 것을 보였습니다.
- 본 코드를 실행하기 위한 필요 라이브러리는 requirements.txt에 있습니다.



## 구성
  ### human_lifelog_mil_pytorch
  - 본 연구에서 사용한 모달 주요도를 추출하는 Multiple Instance Learning model의 structure과 그룹별 라이프로그 예측 모델 structure, 그 외에 실험에 필요한 기본적인 코드들이 있는 디렉토리입니다.
  
### Preprocessing
  - Time_series_to_df.ipynb
  - Time-Series Data 전처리 하는 과정을 ipynb 파일로 정리하였습니다.
  - 2019년 유저별 데이터를 모델에 맞게 DataFrame으로 변환해주는 전처리 코드입니다.

### Visualization
  - 실험 결과를 Visualization하는 코드입니다. 

### Model_training_predict.py
  - 만들어진 그룹을 통해 각 그룹별 Tension 라벨을 예측하는 모델을 학습시키는 코드와, 전체 데이터로 Tension 라벨을 예측하는 코드입니다.


## Dataset

### ETRI Human Life log Dataset
you can download it from [ETRI_Dataset](https://nanum.etri.re.kr/share/schung1/ETRILifelogDataset2020?lang=ko_KR)

## 시각화
```
├──dataset/etri(데이터 다운로드 필요)
├──human_liffelog_mil_pytorch
├── Preprocessing
├── Visualization
│  ├── Modal_importance_analysis.ipynb (해당 파일에서 추출한 시각화 자료들을 추출하였습니다.)
```

