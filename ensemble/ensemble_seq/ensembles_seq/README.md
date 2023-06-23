## Ensemble Seq
seq column을 추가하여 Hard Voting을 변형한 앙상블 기법
 
`input -> files`
```bash
python ensemble_seq.py --target1='top10/EASE_sota.csv' --target2='top10/SASRec_sota.csv' -n 'ensemble.csv'
```

`input -> directory`
```bash
python ensemble_seq.py --input_dir='/opt/ml/input/code/level2_movierecommendation-recsys-04/output/top10' -n 'ensemble.csv'
```