from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np
import math

def getF1Score(label_list, predict_list):
    sum_recall = 0
    average_recall = 0

    sum_precision = 0
    average_precision = 0

    # average_recall 구하기
    temp_answer_list = []
    temp_predict_list = []

    for i in range(len(set(label_list))):
        for j in range(len(predict_list)):
            if label_list[j] != i:
                temp_answer_list.append(0)
    
            else:
                temp_answer_list.append(1)

            if predict_list[j] != i:
                temp_predict_list.append(0)
    
            else:
                temp_predict_list.append(1)

        temp_recall = recall_score(temp_answer_list, temp_predict_list)
        temp_precision = precision_score(temp_answer_list, temp_predict_list)
  
        sum_recall += temp_recall
        sum_precision += temp_precision

    average_recall = sum_recall / len(set(label_list))
    average_precision = sum_precision / len(set(label_list))

    f1_sc = 2 * (average_precision * average_recall) / (average_precision + average_recall)

    if np.isnan(f1_sc):
        return 0

    return f1_sc