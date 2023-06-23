# Sequential Models

## BERT4Rec
### BERT4Rec 모델 모듈화
- dataset.py : 모델 train 코드에 활용할 데이터 셋 클래스 정의
- models.py : BERT4Rec architecture 정의
- preprocessing.py : train과 inference에 필요한 변수들 가져오는 함수 정의
- train.py : 모델 학습 코드
- utils.py : 기타 학습 환경 관련 코드
### config를 yaml파일로 관리
- configs/Ver_0_1_0.yaml : 예시 config
### 인퍼런스 코드 작성
- inference.py : 모델 인퍼런스 코드
### 모델 훈련
```shell
python train.py -c {config 버전 ex) 0_1_0}
```
### 인퍼런스
```shell
python inference.py -mp {모델 pth 파일 경로} -c {config 버전 ex) 0_1_0} -k {top-k k값}
```

## SASRec
SASRec 모델 학습 및 추론
 
`input -> 없음`
```bash
python setup.py
```

`input -> step, configs`
```bash
python run.py -s {모델 학습 및 추론 선택:=["pre_train", "train", "inference"]} --hyperparameters {모델 하이퍼파라미터}
```