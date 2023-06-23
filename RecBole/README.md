## RecBole
RecBole 라이브러리를 활용하여 예측
 
`input -> 없음`
```bash
python setup.py
```

`input -> model_name, config_ver`
```bash
python train.py -m {RecBole에 존재하는 사용 모델 이름:=RecVAE} -c {config 버전:=0}
```

`input -> saved_model_path, config_ver`
```bash
python inference.py -m {모델 학습 후 saved 폴더에 저장된 모델 경로} -c {config 버전:=0}
```




