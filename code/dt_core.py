# version 1.0
import math
from typing import List
from anytree import Node

import dt_global as dtg
import dt_provided as dt_provided


def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """
    split_points = set([])
    setList = []
    possible_split = set([])
    index = 0

    
    for i in range(len(dtg.feature_names)):
        if dtg.feature_names[i] == feature:
            break
        index = index + 1
    
    if examples != None:
        examples.sort(key = lambda x: x[index])
        for i in range(len(examples) -1):
            possible_split.add((examples[i][index] + examples[i+1][index])/2)

    for i in range(dtg.num_label_values):
        setX = list(filter(lambda x: x[dtg.label_index] == i, examples))
        setList.append(setX)

    for i in range(len(setList) - 1):
        #compare adjacent sets
        #ex: if there are two labels, 0 and 1, then we will compare all items of 0 with all items of 1
        for j in setList[i]:
            for k in setList[i+1]:
                if (((j[index] + k[index])/2) in possible_split and not math.isclose(j[index], k[index], abs_tol=1e-8)):
                        split_points.add((j[index] + k[index])/2)

    return list(split_points)


def entropy(prob_dist):
    result = 0
    for j in prob_dist:
        if j != 0:
            result += -(j * math.log2(j))
    return result

def choose_feature_split(examples: List, features: List[str]) -> (str, float, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None, -1, and -inf.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature, the best split value, the max expected information gain
    :rtype: str, float, float
    """   
    entropy_all = []
    best_splits = []
    valid_split = False
    n = len(examples)
    items_of_feature = []
    expected_info_gain_before = 0
    for j in range(dtg.num_label_values):
        num_items = 0
        for k in examples:
            if k[dtg.label_index] == j:
                num_items = num_items + 1
        items_of_feature.append(num_items/n) # will produce a probability distribution
    
    #entropy: -(j * np.log2(j))
    
    expected_info_gain_before = entropy(items_of_feature)

    resulting_gain = math.inf
    best_index = 0

    for i in features:
        #get the splits for each feature
        feature_split = sorted(get_splits(examples, i))

        max_entropy = math.inf
        current_best_split = 0.0

        for j in feature_split: #all split points for feature i
            
            valid_split = True
            less_than_j = []
            greq_than_j = []
            
            #need to get num items less than j, num items greater than / equal to j
            for x in examples:
                if dt_provided.less_than_or_equal_to(x[dtg.feature_names.index(i)], j):
                    less_than_j.append(x)
                else:
                    greq_than_j.append(x)
                

            less_than_j_dist = [0] * dtg.num_label_values
            greq_than_j_dist = [0] * dtg.num_label_values

            #for all items less than the split value, get the number of items corresponding to each label
            
            if len(less_than_j) > 0:
                for l in less_than_j:
                    less_than_j_dist[l[dtg.label_index]] += 1
                less_than_j_dist[:] = [x / len(less_than_j) for x in less_than_j_dist]
                #for all items greater than/equal to the split value, get the number of items corresponding to each label
            if len(greq_than_j) > 0:
                for l in greq_than_j:
                    greq_than_j_dist[l[dtg.label_index]] += 1
                greq_than_j_dist[:] = [x / len(greq_than_j) for x in greq_than_j_dist]

            ltj_entropy = entropy(less_than_j_dist)
            ltj_entropy = ltj_entropy * (len(less_than_j) / n)

            gtj_entropy = entropy(greq_than_j_dist)
            gtj_entropy = gtj_entropy * (len(greq_than_j) / n)



            max_entropy_before = max_entropy

            max_entropy = min(max_entropy, ltj_entropy + gtj_entropy)

            if max_entropy_before != max_entropy:
                current_best_split = j

        entropy_all.append(max_entropy)
        best_splits.append(current_best_split)

    if not valid_split:
        return None, -1, -math.inf
    for i in range(len(entropy_all)):
        temp = resulting_gain
        resulting_gain = min(entropy_all[i] - expected_info_gain_before, resulting_gain)

        if temp != resulting_gain:
            best_index = i

    return dtg.feature_names[best_index], best_splits[best_index], resulting_gain



def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """ 
    list_leseq = []
    list_greater = []
    for x in examples:
        if dt_provided.less_than_or_equal_to(x[dtg.feature_names.index(feature)], split):
            list_leseq.append(x)
        else:
            list_greater.append(x)
        
    return list_leseq, list_greater


def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """
    if len(examples) == 0:
        #make a leaf, need to get the majority decision at parent
        majority_label = cur_node.parent.decision
        cur_node.name = "/leaf_no_examples"
        cur_node.decision = majority_label
        return
    elif cur_node.depth == max_depth:
        #make a leaf, need to get the majority decision
        tally = [0] * dtg.num_label_values
        for x in examples:
            tally[x[dtg.label_index]] += 1
        majority_decision = -1
        majority_label = -1
        for i in range(len(tally)):
            if tally[i] > majority_decision:
                majority_decision = tally[i]
                majority_label = i
        cur_node.name = "/leaf_no_depth"
        cur_node.decision = majority_label
        return
    else:
        #check if all examples in the same class:
        first_class = -1
        if len(examples) > 0:
            first_class = examples[0][dtg.label_index]
        for i in examples:
            if i[dtg.label_index] != first_class:
                first_class = -1
                break
        best_feature, best_split, best_info_gain = choose_feature_split(examples, features)

        #all examples have the same label
        if first_class != -1:
            cur_node.name = "/leaf_same_class"
            cur_node.decision = first_class
            return

        #no features to split at this current node
        if best_feature == None:
            tally = [0] * dtg.num_label_values
            for x in examples:
                tally[x[dtg.label_index]] += 1
            majority_decision = -1
            majority_label = -1
            for i in range(len(tally)):
                if tally[i] > majority_decision:
                    majority_decision = tally[i]
                    majority_label = i
            cur_node.name = "/leaf_no_features"
            cur_node.decision = majority_label
            return
            
        #make children
        else:
            tally = [0] * dtg.num_label_values
            for x in examples:
                tally[x[dtg.label_index]] += 1
            majority_decision = -1
            majority_label = -1
            for i in range(len(tally)):
                if tally[i] > majority_decision:
                    majority_decision = tally[i]
                    majority_label = i
            cur_node.decision = majority_label
            cur_node.feature = best_feature
            cur_node.split = best_split
            cur_node.info_gain = best_info_gain
            left_child, right_child = split_examples(examples, best_feature, best_split)
            left_node: Node = Node("/l", parent=cur_node, feature=None, split=None, info_gain=None, decision=None)
            right_node: Node = Node("/r", parent=cur_node, feature=None, split=None, info_gain=None, decision=None)
            split_node(left_node, left_child, features, max_depth)
            split_node(right_node, right_child, features, max_depth)


def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    
    cur_node: Node = Node("root", parent=None, feature=None, split=None, decision=None, info_gain = None)
    if len(examples) == 0:
        return None
    else:
        split_node(cur_node, examples, features, max_depth)
        return cur_node


def predict(cur_node: Node, example, max_depth=math.inf) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the decision for the given example
    :rtype: int
    """ 

    if cur_node.split == None or cur_node.depth == max_depth:
        return cur_node.decision
    node_feature = cur_node.feature
    feature_index = dtg.feature_names.index(node_feature)
    #recurse on the left
    if dt_provided.less_than_or_equal_to(example[feature_index], cur_node.split):
        return predict(cur_node.children[0], example, max_depth)
    else:
        return predict(cur_node.children[1], example, max_depth)



def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth,
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    num_correct = 0
    for i in examples:
        prediction = (predict(cur_node, i, max_depth))
        if prediction == i[dtg.label_index]:
            num_correct+=1
    if len(examples) > 0:
        return num_correct / len(examples)
    else: 
        return 0

def post_prune(cur_node: Node, min_info_gain: float):
    """
    Given a tree with cur_node as the root, and the minimum information gain,
    post prunes the tree using the minimum information gain criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the information gain at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the information gain at every leaf parent is greater than
    or equal to the pre-defined value of the minimum information gain.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_info_gain: the minimum information gain
    :type min_info_gain: float
    """
    no_children = False
    #if a node has descendants
    if len(cur_node.children) > 0:
        #both children are leaves
        if (cur_node.children[0].split == None and cur_node.children[1].split == None):
            no_children = True
        #one child is a leaf
        elif (cur_node.children[0].split == None and cur_node.children[1].split != None):
            post_prune(cur_node.children[1], min_info_gain)
        elif (cur_node.children[0].split != None and cur_node.children[1].split == None):
            post_prune(cur_node.children[0], min_info_gain)
        #both children are not leaves
        else:
            post_prune(cur_node.children[0], min_info_gain)
            post_prune(cur_node.children[1], min_info_gain)
        if dt_provided.less_than(-1 * cur_node.info_gain, min_info_gain):
                cur_node.children[0].parent = None
                cur_node.children[0].parent = None
                cur_node.split = None
    #if a node has no descendants, its a leaf
    else:
        return
