# ASR project barebones

Report:
logs:
info.logs - best model train logs

## Installation guide

```shell
pip install -r ./requirements.txt
```

Best Model:
```shell
curl -c ./temp.txt -s -L "https://drive.google.com/uc?export=download&id=1gO-rlTsz0D0WveD82MABw71tpk2WrElK" > /dev/null
curl -Lb ./temp.txt "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie.txt`&id=1gO-rlTsz0D0WveD82MABw71tpk2WrElK" -o "model_best/model_best.pth"
```

Google Drive Link: https://drive.google.com/file/d/1gO-rlTsz0D0WveD82MABw71tpk2WrElK/view?usp=sharing

To reproduce: Train 15 epoches with kaggle_train_no_abs.json

To test:
```shell
python3 test.py -c best_model/config.json -r best_model/model_best.pth -t test_data -o output.json
```
