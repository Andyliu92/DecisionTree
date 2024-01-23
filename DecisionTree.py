"""
Authors: Ashwani Kashyap, Anshul Pardhi
"""

import math
import numpy as np
from typing import Union, Tuple
from pathlib import Path
from copy import deepcopy
import re

scriptDir = Path(__file__)


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])


#######
# Demo:
# unique_vals(training_data, 0)
# unique_vals(training_data, 1)
#######


def class_counts(rows: np.ndarray) -> dict:
    """Counts the number of each type of example in a dataset."""
    unique_values, counts = np.unique(rows[:, -1], return_counts=True)
    label2count = {unique_values[i]: counts[i] for i in range(len(unique_values))}
    return label2count


#######
# Demo:
# class_counts(training_data)
#######


def max_label(dict: dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label


def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


#######
# Demo:
# is_numeric(7)
# is_numeric("Red")
#######


class Question:
    """A Question is used to partition a dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    question. See the demo below.
    """

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val <= self.value
        else:
            assert 0, "This version of code deals with pure numerical value"
            return val == self.value

    def treeTextQuestion(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "<="
        if not is_numeric(self.value):
            raise NotImplementedError(
                "all values in the feature should be numeric when adding variation."
            )
        return "%s %s %s" % (self.header[self.column], condition, str(self.value))

    def treeTextInverseQuestion(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = ">"
        if not is_numeric(self.value):
            raise NotImplementedError(
                "all values in the feature should be numeric when adding variation."
            )
        return "%s %s %s" % (self.header[self.column], condition, str(self.value))

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = "<="
        return "Is %s %s %s?" % (self.header[self.column], condition, str(self.value))

    def partition(self, rows: np.ndarray):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        # rows = np.array(rows)
        mask = rows[:, self.column] <= self.value
        true_rows = rows[mask]
        false_rows = rows[~mask]
        return true_rows, false_rows


def gini(rows):
    """Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


## TODO: Step 3
def entropy(rows: np.ndarray):
    assert len(rows.shape) == 2, "error: this function only works with 2-d array!"
    # compute the entropy.
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(rows.shape[0])
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1 * avg_entropy


def info_gain(left, right, current_uncertainty):
    """Information Gain.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))

    ## TODO: Step 3, Use Entropy in place of Gini
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows, header):
    """Find the best question to ask by iterating over every feature / value
    and calculating the information gain."""
    # raise DeprecationWarning("This function should not be called with variation")
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value
            question = Question(col, val, header)

            gain = __calculate_gain(question, rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


def find_best_split_var(
    rows: np.ndarray, header, config: dict
) -> Tuple[float, Question]:
    """
    Find the best question to ask by iterating over every feature / value
    and calculating the information gain.
    This function considers that some variation may happen in the value and
    train the tree with variation.
    """
    assert config[
        "hasWeightVar"
    ], 'to use this function for training, "hasWeightVar" in config must be true!'
    best_gain = 0  # keep track of the best information gain
    best_question = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = rows.shape[1] - 1  # number of columns
    sampleTimes = config["weightVar"]["sampleTimes"]
    varStdDev = config["weightVar"]["stdDev"]

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])  # unique values in the column

        sortedValues = sorted(values)
        midPointValues = [
            (sortedValues[i] + sortedValues[i + 1]) / 2
            for i in range(len(sortedValues) - 1)
        ]

        for val in midPointValues:  # for each value
            gain = 0
            for i in range(sampleTimes):
                valWithVar = val + np.random.normal(0, varStdDev)
                question = Question(col, valWithVar, header)

                gain += __calculate_gain(question, rows, current_uncertainty)

            # You actually can use '>' instead of '>=' here
            # but I wanted the tree to look a certain way for our
            # toy dataset.
            gain /= sampleTimes
            if gain >= best_gain:
                best_gain, best_question = gain, Question(col, val, header)
    return best_gain, best_question


def __calculate_gain(question: Question, rows: np.ndarray, currentUncertainty) -> float:
    # try splitting the dataset
    true_rows, false_rows = question.partition(rows)

    # Skip this split if it doesn't divide the
    # dataset.
    if len(true_rows) == 0 or len(false_rows) == 0:
        return 0

    # Calculate the information gain from this split
    gain = info_gain(true_rows, false_rows, currentUncertainty)
    return gain


## TODO: Step 2
class Leaf:
    """A Leaf node classifies data.

    This holds a dictionary of class (e.g., "Apple") -> number of times
    it appears in the rows from the training data that reach this leaf.
    """

    def __init__(self, rows, id, depth, type: str):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth
        self.type = type


## TODO: Step 1
class Decision_Node:
    """A Decision Node asks a question.

    This holds a reference to the question, and to the two child nodes.
    """

    def __init__(
        self, question: Question, true_branch, false_branch, depth, id, rows, type: str
    ):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows
        self.type = type


## TODO: Step 3
def build_tree(
    rows: np.ndarray, header, config: dict, depth=0, id=0
) -> Union[Decision_Node, Leaf]:
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """
    # depth = 0
    # Try partitioing the dataset on each of the unique attribute,
    # calculate the information gain,
    # and return the question that produces the highest gain.

    # print(rows)

    if config["hasWeightVar"]:
        gain, question = find_best_split_var(rows, header, config)
    else:
        gain, question = find_best_split(rows, header)

    # Base case: no further info gain
    # Since we can ask no further questions,
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, id, depth, "original")

    # If we reach here, we have found a useful feature / value
    # to partition on.
    # nodeLst.append(id)
    true_rows, false_rows = question.partition(rows)

    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header, config, depth + 1, 2 * id + 2)

    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header, config, depth + 1, 2 * id + 1)

    # Return a Question node.
    # This records the best feature / value to ask at this point,
    # as well as the branches to follow
    # depending on on the answer.
    return Decision_Node(
        question, true_branch, false_branch, depth, id, rows, "original"
    )


## TODO: Step 8 - already done for you
def prune_tree(node, prunedList):
    """Builds the tree.

    Rules of recursion: 1) Believe that it works. 2) Start by checking
    for the base case (no further information gain). 3) Prepare for
    giant stack traces.
    """

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node
    # If we reach a pruned node, make that node a leaf node and return. Since it becomes a leaf node, the nodes
    # below it are automatically not considered
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth, "original")

    # Call this function recursively on the true branch
    node.true_branch = prune_tree(node.true_branch, prunedList)

    # Call this function recursively on the false branch
    node.false_branch = prune_tree(node.false_branch, prunedList)

    return node


## TODO: Step 6
def classify(row, node):
    """See the 'rules of recursion' above."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        return node.predicted_label

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


## TODO: Step 4
def print_tree(node, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(
            spacing
            + "Leaf id: "
            + str(node.id)
            + " Predictions: "
            + str(node.predictions)
            + " Label Class: "
            + str(node.predicted_label)
        )
        return

    # Print the question at this node
    print(
        spacing
        + str(node.question)
        + " id: "
        + str(node.id)
        + " depth: "
        + str(node.depth)
    )

    # Call this function recursively on the true branch
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")


def exportTreeText(node: Decision_Node, outputFile, spacing=""):
    """World's most elegant tree printing function."""

    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        outputFile.write(
            spacing + "|---" + " class: " + str(node.predicted_label) + "\n"
        )
        return

    # Print the question at this node
    outputFile.write(spacing + "|---" + str(node.question.treeTextQuestion()) + "\n")
    # Call this function recursively on the true branch
    exportTreeText(node.true_branch, outputFile, spacing + "|   ")

    outputFile.write(
        spacing + "|---" + str(node.question.treeTextInverseQuestion()) + "\n"
    )
    # Call this function recursively on the false branch
    exportTreeText(node.false_branch, outputFile, spacing + "|   ")


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


## TODO: Step 5
def getLeafNodes(node, leafNodes=[]):
    # Base case
    if isinstance(node, Leaf):
        leafNodes.append(node)
        return

    # Recursive right call for true values
    getLeafNodes(node.true_branch, leafNodes)

    # Recursive left call for false values
    getLeafNodes(node.false_branch, leafNodes)

    return leafNodes


def getInnerNodes(node, innerNodes=[]):
    # Base case
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)

    # Recursive right call for true values
    getInnerNodes(node.true_branch, innerNodes)

    # Recursive left call for false values
    getInnerNodes(node.false_branch, innerNodes)

    return innerNodes


## TODO: Step 6
def computeAccuracy(rows, node):
    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        # last entry of the column is the actual label
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy / count, 2)


def copyTree(node: Union[Decision_Node, Leaf]) -> Union[Decision_Node, Leaf]:
    if isinstance(node, Leaf):
        result = deepcopy(node)
        result.type = "copied"
        return result
    elif isinstance(node, Decision_Node):
        return Decision_Node(
            node.question,
            copyTree(node.true_branch),
            copyTree(node.false_branch),
            node.depth,
            node.id,
            node.rows,
            "copied",
        )
    else:
        raise NotImplementedError("unexpected node class")


# Parse the tree structure text into a nested dictionary
def parseTreeStructure(
    text: str,
) -> Union[Decision_Node, Leaf]:
    """
    this function takes a tree text and returns the parsed tree structure and all leaf nodes, and all feature ids, class ids, and thresholds used in the tree.
    """
    lines = text.strip().split("\n")

    leafNodes = []
    featureIDs = []
    classIDs = []
    thresholds = []
    (
        treeDict,
        subTreeEndLineID,
        leafNodes,
        featureIDs,
        classIDs,
        thresholds,
    ) = __parseSubTree(lines, 0, leafNodes, featureIDs, classIDs, thresholds)
    assert (
        subTreeEndLineID == len(lines) - 1
    ), "ERROR: the tree is not completely parsed!"

    return treeDict


__nodeUID = 0  # add an uid to make each node unique


def __parseSubTree(
    lines: list[str],
    lineID: int,
    # parentNode: Union[dict, None],
    leafNodes: list[dict],
    featureIDs: list[int],
    classIDs: list[int],
    thresholds: list[float],
) -> Tuple[
    Union[Leaf, Decision_Node], int, list[dict], list[int], list[int], list[float]
]:
    """
    this function takes all lines of a tree, the line id where the subtree to be parsed starts, and the pointer to parentNode.
    this function returns the parsed sub tree and the line id where the subtree ends(not the id where the following structure of tree starts!).
    """
    global __nodeUID
    if "class:" in lines[lineID]:  # this subtree is a leaf node
        classID = re.search("[0-9]+.?0?", lines[lineID]).group()
        classID = int(classID.strip().split(".")[0])
        # leafNode = {"class": classID, "parent": parentNode, "uid": __nodeUID}
        leafNode = Leaf(np.array([[0]]), __nodeUID, 0, "loaded")
        leafNode.predicted_label = classID
        __nodeUID += 1
        leafNodes.append(leafNode)
        if classID not in classIDs:
            classIDs.append(classID)
        return (leafNode, lineID, leafNodes, featureIDs, classIDs, thresholds)
    elif re.search(
        "feature_[0-9]+ <=", lines[lineID]
    ):  # is a stem node, parse recursively
        # the following code assumes that le and ge nodes both exists!
        # stemNode = {
        #     "featureID": int,
        #     "threshold": float,
        #     "leNode": dict,  # less or equal. e.g. feature 3 <= 4
        #     "gtNode": dict,  # greater than.  e.g. feature 3 >  4
        #     "parent": Union[None, dict],
        #     "uid": __nodeUID,
        # }
        __nodeUID += 1
        matchStr = re.search("feature_[0-9]+", lines[lineID]).group()
        featureID = int(matchStr.split("_")[1])
        if featureID not in featureIDs:
            featureIDs.append(featureID)
        matchStr = re.search("<=[ ]+[-]?[0-9]+.[0-9]+", lines[lineID]).group()
        threshold = float(matchStr.split(" ")[-1])
        thresholds.append(threshold)
        leNode, endLineId, leafNodes, featureIDs, classIDs, thresholds = __parseSubTree(
            lines, lineID + 1, leafNodes, featureIDs, classIDs, thresholds
        )
        assert re.search(
            f"feature_{featureID} >", lines[endLineId + 1]
        ), "ERROR: the gt branch does not follow the end of last sub tree."
        gtNode, endLineId, leafNodes, featureIDs, classIDs, thresholds = __parseSubTree(
            lines, endLineId + 2, leafNodes, featureIDs, classIDs, thresholds
        )
        # stemNode["featureID"] = featureID
        # stemNode["threshold"] = threshold
        # stemNode["leNode"] = leNode
        # stemNode["gtNode"] = gtNode

        question = Question(featureID, threshold, f"feature_{featureID}")

        stemNode = Decision_Node(question, leNode, gtNode, 0, __nodeUID, np.array([[0]]), "loaded")

        return stemNode, endLineId, leafNodes, featureIDs, classIDs, thresholds
    else:
        assert not re.search(
            "truncated", lines[lineID]
        ), "Tree too deep. Some branch is truncated in tree text"
        raise NotImplementedError
