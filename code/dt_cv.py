# version 1.0
from typing import List

import dt_global as dtg
from dt_core import *
from anytree import RenderTree


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """
    avg_training_accuracies = []
    avg_validation_accuracies = []
    for i in value_list:
        prediction_accuracies_training = []
        prediction_accuracies_validation = []
        
        for j in range(len(folds)):
            training_set = []
            validation_set = []
            validation_set = folds[j]
            for k in range(len(folds)):
                if k != j:
                    training_set = training_set + folds[k]
            input_feature_names = dtg.feature_names[:-1]

            tree = learn_dt(training_set, input_feature_names, i)

            training_accuracy = get_prediction_accuracy(tree, training_set, i)
            validation_accuracy = get_prediction_accuracy(tree, validation_set, i)
            prediction_accuracies_training.append(training_accuracy)
            prediction_accuracies_validation.append(validation_accuracy)
        avg_train_result = sum(prediction_accuracies_training)/len(prediction_accuracies_training)
        avg_valid_result = sum(prediction_accuracies_validation)/len(prediction_accuracies_validation)
        avg_training_accuracies.append(avg_train_result)
        avg_validation_accuracies.append(avg_valid_result)
    
    return avg_training_accuracies, avg_validation_accuracies


def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 

    avg_training_accuracies = []
    avg_validation_accuracies = []
    for i in value_list:
        prediction_accuracies_training = []
        prediction_accuracies_validation = []
        
        for j in range(len(folds)):
            training_set = []
            validation_set = []
            validation_set = folds[j]
            for k in range(len(folds)):
                if k != j:
                    training_set = training_set + folds[k]
            input_feature_names = dtg.feature_names[:-1]
            print(training_set)
            print("---")
            
            tree = learn_dt(training_set, input_feature_names)
            post_prune(tree, i)

            training_accuracy = get_prediction_accuracy(tree, training_set)
            print(training_accuracy)
            validation_accuracy = get_prediction_accuracy(tree, validation_set)
            prediction_accuracies_training.append(training_accuracy)
            prediction_accuracies_validation.append(validation_accuracy)
        avg_train_result = sum(prediction_accuracies_training)/len(prediction_accuracies_training)
        avg_valid_result = sum(prediction_accuracies_validation)/len(prediction_accuracies_validation)
        avg_training_accuracies.append(avg_train_result)
        avg_validation_accuracies.append(avg_valid_result)
    
    return avg_training_accuracies, avg_validation_accuracies
