import numpy as np

def multiclass_accuracy(prediction, ground_truth):
    correct = np.where(prediction == ground_truth)[0]
    accuracy = correct.shape[0]/prediction.shape[0]
    
    return accuracy
