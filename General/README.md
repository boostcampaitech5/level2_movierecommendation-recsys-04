## General
General-recommendation 모델 중 train data를 분리하지 않고 예측
 
`input -> model_name, config_ver`
```bash
python run.py -m {모델 이름:=EASE} -c {config 버전:=1}
```