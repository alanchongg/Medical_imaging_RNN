# Medical_imaging_RNN
Using Recurrent Neural Network (RNN) done using PyTorch to classifiy lung diseases  
soruce of dataset: https://www.kaggle.com/datasets/fatemehmehrparvar/lung-disease/data 

The downloaded dataset can be placed in the same folder, using the prep.py will split the data into 3 different sets: trianing, validation, testing.  
The main.py file can be used to change the hyperparameters of the training process prior to running it.  
Training details will be saved iin a txt file which can be used by visualisation.py to view training and validation loss, training and validation F1 score.
Trained model will be saved in the Rnnstate.pt which can be used by the tk.py which provides a intuative and funcational GUI for making prediction.  
This project includes the comparison for 3 different varient of RNN which includes: vanilla RNN, BiDirectional RNN & LSTM
The final model used to prediction can be changed by changing the loadstate in tk.py

## Verdict
This was a proof of concept of using RNN to identify any long-haul conditions in medical imaging tasks, justifiable and unbiased result achieved a test F1_score of around 0.70.
Results proofed the possibility of RNN in medical imaging task, however project was discontinued due to better results from using Convolutional Neural Network.
