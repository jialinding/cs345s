Homework for CS 345S

Built upon work from https://github.com/dennybritz/cnn-text-classification-tf

A CNN for topic categorization created using TensorFlow, trained and tested on 
two topics (alt.atheism and comp.graphics) from the 20-newsgroups dataset.

Classification accuracy on the test set is 61%. Easy improvements such as increasing
epochs, embedding dimensions, filter sies, or filter number; adding dropout; and using
better word embeddings could improve accuracy. However, since this was primarily a
didactic exercise, no such attempts at improvement were made.