How to run this project?

1. Upload the whole baseline folder to coli sever.
2. Install the requirements.txt.
3. Unzip the CCPC.zip
4. run the run.sh

You will get:
1. A best model with the lowest validation loss
2. A log file with loss information
3. A checkpoint file
4. Plots in your wandb account

Current observations:
The best validation loss(about 5.0) is still too high to use. 

Note: 
The decode function in train.py uses ground true targets, which is not correct for generation, I just use it 
to indicate how well the trainning is. But the greedy_decode in test.py is a correct example for generation, which is mainly to show you how to restore the model.

Welcome to add more features/better model design/suggestions on fine-tuning/corrections.
Contact me if anything.
