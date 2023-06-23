## valid_sampler
랜덤 시드를 활용하여 validation set 구축
 
`input -> n_iter, start_seed`
```bash
python sampler.py -n {반복 횟수:=100} -s {시작 랜덤 시드:=0}
```

`input -> start_seed, end_seed`
```bash
python get_sample.py -s {시작 랜덤 시드:=0} -e {종료 랜덤 시드:=2282}
```