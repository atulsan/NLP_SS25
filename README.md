# NLP_SS25
Track A (Multi-label Emotion Detection) : 2025 SemEval Shared Task 11

# Steps to run the python file
1. Download the best model from hessen box and unzip the file with the same folder, so best_model folder is present in the root directory
    https://hessenbox.uni-marburg.de/getlink/fiKPpZCajLTYqUkxRyiB5n/best_model.zip
    (click download button)

2. Install all dependencies
```shell
pip install -r requirements.txt
```
3. For training the model (can be skipped if model is downloaded)
```shell
python main.py train
```
4. For training the model (can be skipped if model is downloaded)
```shell
python main.py --input_csv input_file.csv predict
``` 
This will create a predictions.csv file which can be used for evaluation

# F1 score
77% Accuracy
```shell
{'eval_loss': 0.8552697896957397, 'eval_f1': 0.7739283617146213, 'eval_runtime': 12.2294, 'eval_samples_per_second': 45.301, 'eval_steps_per_second': 1.472, 'epoch': 10.0}
```

