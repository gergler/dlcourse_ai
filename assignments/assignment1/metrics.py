import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    true_positive, false_positive, false_negative = 0, 0, 0
    for i in zip(prediction, ground_truth):
        if i == (1, 1):
            true_positive += 1
        elif i == (0, 1):   
            false_negative += 1
        elif i == (1, 0):
            false_positive += 1
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    
    correct = np.where(prediction == ground_truth)[0]
    accuracy = correct.shape[0]/prediction.shape[0]
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    correct = np.where(prediction == ground_truth)[0]
    accuracy = correct.shape[0]/prediction.shape[0]
    return accuracy
