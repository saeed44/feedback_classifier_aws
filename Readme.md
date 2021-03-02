# A webapp to classify customer feedback.

This is a containerized webapp (using Docker). 

To run the application locally, use docker `build` and `run` commands. 

Here is a live demo of the application: http://3.136.83.118:5000

When testing samples using this model, note that the model has been trained on a data set that is collected from non-native english speakers. While the performance on the test set is very good (accuracy~95%), it might not have the best performance on more sophisticated use of the english words.
