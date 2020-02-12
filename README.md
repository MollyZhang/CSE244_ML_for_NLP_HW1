# CSE244_ML_for_NLP
Code for CSE244 winter 2020 homework 1: Relation Extraction From Text

### Staring script:
run.ipynb or run-bert.ipynb

### Folder structure:
- model_utils.py

This file contains all models I have tried
- data_utils.py

This contains preparing data with torchtext, for both ngram format and also word embedding format. This file also include wrappers which generate data batches for training. 
- train_utils.py

This file contains the function used for training and calculating losses during training
- evaluation.py

This file contains calculation of f1 scores in various ways
- submission.py

This file generate submission file for kaggle. It also includes an Ensemble class to combine results of multiple models into one final result. It also includes basic error anaylysis. 
- preprocessing.ipynb

This is the notebook for preprocessing: train, validation, test splitting. Converting label text to numerical categories. Among with many minute things I tried. 

- movie_and_person_names.ipynb 

This is the notebook specificially for one type of feature engineering: processing movie and people names into phrases.

- run.ipynb and run-bert.ipynb
This should be the starting point. This is where I import all other functions from \*.py files and get results.





