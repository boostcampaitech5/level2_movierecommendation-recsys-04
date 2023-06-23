## Ensemble Base
Hard Voting Soft Voting 앙상블 기법
 
`Hard Voting`
```bash
python run_voting.py --strategy 'hard' --file_path './submissions/ensembles/'
python run_voting.py --files ease,sasrec --strategy 'hard'
```

`Soft Voting`
```bash
python run_voting.py --files ease,sasrec --strategy 'soft' --weight 0.3,0.7