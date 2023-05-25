# mipt food retail review classification
User Review Classification in Food Retail With Small Amount of Training Data and Many Classes

See .pdf with work description placed in root dir
 
Dataset is placed in scrub dir

Brief notebook contents:

nb1 contains code that takes initial dataset as input and implies cascade data preprocessings, saves result to file

nb2 contains GBDT classifier construction, and calculation of increment of every step of data preprocessing to final F1 score metric, in order to find best preprocessing technique

nb3 contains construction and fine-tuning on BERT for sequence classification with plots and different checpoints and different input data (raw or preprocessed)

nb4 contains construction of custom BERT and its fine-tuning, with steps analogical to nb3 in order to compare them. Also it scores the result with custom metric, denoted as positional recall, in order to compare with result of nb6

nb5 contains code to transform initial multi-class task statement to multi-label. It creates new dataset and saves it to file

nb6 contains code, that uses best model, received on previous steps, applied to multi-label transformed task, and scores it with positional recall, in order to compare with nb4
