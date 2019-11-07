# COMP551_ImageClassification
We use .gitignore to ignore all the huge data files.
Please makesure all the 3 files test_max_x train_max_x train_may_y.csv are all locally stored in your pycharm folder (COMP551_ImageCLassification)

Hi, there is a new file clean_Data. After you run it you should get a new dataset without backgrounds. 
I would like to ask you to still build your models with the normal Dataset. We will try the cleaned one afterwards. 
Sometimes it does weird things, but in this case I really think it helps. 

Idea to be implemented:

Use a pre-trained MNIST model (from internet, not implemented by us) to predict our preprocessed data.

That is, predict the three digits from one image and take the max digit predicted as the predicted label (y)

Since we are given the train_labels, we can then compare the given train_labels with the predicted labels by us to find the accuracy of the training dataset

If the accuracy is good, we can apply the same strategy on test dataset
