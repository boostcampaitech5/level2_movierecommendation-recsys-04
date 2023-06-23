## Ensemble Rank
등수별 가중치를 적용한 soft voting(모델별 가중치)
 
`input -> files`
```bash
python ensemble.py --strategy 'hard' --file_path './submissions/rank_files/' 
```

`input -> weights`
```bash
python ensemble.py --rank_weight '0.7,0.3' --model_weight '0.1065,0.0944,0.1349,0.1551,0.1386,0.0754,0.1614,0.1405'
```
