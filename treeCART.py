import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


class DecisionTree(object):
    """
    Class to create decision tree model (CART)
    """

    def __init__(self, _max_depth, _min_splits):
        self.max_depth = _max_depth
        self.min_splits = _min_splits

    def fit(self, _feature, _label):
        """
        :param _feature:
        :param _label:
        :return:
        """
        self.feature = _feature
        self.label = _label
        self.train_data = np.column_stack((self.feature, self.label))
        self.build_tree()

    def compute_gini_similarity(self, groups, class_labels):
        """
        compute the gini index for the groups and class labels
        :param groups:
        :param class_labels:
        :return:
        """
        num_sample = sum([len(group) for group in groups])
        gini_score = 0

        for group in groups:
            size = float(len(group))

            if size == 0:
                continue
            score = 0.0
            for label in class_labels:
                porportion = (group[:, -1] == label).sum() / size
                score += porportion * porportion
            gini_score += (1.0 - score) * (size / num_sample)

        return gini_score

    def terminal_node(self, _group):
        """
        Function set terminal node as the most common class in the group to make prediction later on
        is an helper function used to mark the leaf node in the tree based on the early stop condition
        or actual stop condition which ever is meet early
        :param _group:
        :return:
        """
        class_labels, count = np.unique(_group[:, -1], return_counts=True)
        return class_labels[np.argmax(count)]

    def split(self, index, val, data):
        """
        split features into two groups based on their values
        :param index:
        :param val:
        :param data:
        :return:
        """
        data_left = np.array([]).reshape(0, self.train_data.shape[1])
        data_right = np.array([]).reshape(0, self.train_data.shape[1])

        for row in data:
            if row[index] <= val:
                data_left = np.vstack((data_left, row))

            if row[index] > val:
                data_right = np.vstack((data_right, row))

        return data_left, data_right

    def best_split(self, data):
        """
        find the best split information using the gini score
        :param data:
        :return best_split result dict:
        """
        class_labels = np.unique(data[:, -1])
        best_index = 999
        best_val = 999
        best_score = 999
        best_groups = None

        for idx in range(data.shape[1] - 1):
            for row in data:
                groups = self.split(idx, row[idx], data)
                gini_score = self.compute_gini_similarity(groups, class_labels)

                if gini_score < best_score:
                    best_index = idx
                    best_val = row[idx]
                    best_score = gini_score
                    best_groups = groups
        result = {}
        result['index'] = best_index
        result['val'] = best_val
        result['groups'] = best_groups
        return result

    def split_branch(self, node, depth):
        """
        recursively split the data and
        check for early stop argument based on self.max_depth and self.min_splits
        - check if left or right groups are empty is yess craete terminal node
        - check if we have reached max_depth early stop condition if yes create terminal node
        - Consider left node, check if the group is too small using min_split condition
            - if yes create terminal node
            - else continue to build the tree
        - same is done to the right side as well.
        else
        :param node:
        :param depth:
        :return:
        """
        left_node, right_node = node['groups']
        del(node['groups'])

        if not isinstance(left_node, np.ndarray) or not isinstance(right_node, np.ndarray):
            node['left'] = self.terminal_node(left_node + right_node)
            node['right'] = self.terminal_node(left_node + right_node)
            return

        if depth >= self.max_depth:
            node['left'] = self.terminal_node(left_node)
            node['right'] = self.terminal_node(right_node)
            return

        if len(left_node) <= self.min_splits:
            node['left'] = self.terminal_node(left_node)
        else:
            node['left'] = self.best_split(left_node)
            self.split_branch(node['left'], depth + 1)

        if len(right_node) <= self.min_splits:
            node['right'] = self.terminal_node(right_node)
        else:
            node['right'] = self.best_split(right_node)
            self.split_branch(node['right'], depth + 1)

    def build_tree(self):
        """
        build tree recursively with help of split_branch function
         - Create a root node
         - call recursive split_branch to build the complete tree
        :return:
        """
        self.root = self.best_split(self.train_data)
        self.split_branch(self.root, 1)
        return self.root

    def _predict(self, node, row):
        """
        Recursively traverse through the tress to determine the
        class of unseen sample data point during prediction
        :param node:
        :param row:
        :return:
        """
        if row[node['index']] < node['val']:
            if isinstance(node['left'], dict):
                return self._predict(node['left'], row)
            else:
                return node['left']

        else:
            if isinstance(node['right'], dict):
                return self._predict(node['right'], row)
            else:
                return node['right']

    def predict(self, test_data):
        """
        predict the set of data point
        :param test_data:
        :return:
        """
        self.predicted_label = np.array([])
        for idx in test_data:
            self.predicted_label = np.append(
                self.predicted_label, self._predict(self.root, idx))

        return self.predicted_label


def accuracy(prediction, actual):
        """
        :param prediction:
        :param actual:
        :return accuaracy:
        Simple function to compute raw accuaracy score quick comparision.
        """
        correct_count = 0
        prediction_len = len(prediction)
        for idx in range(prediction_len):
            if int(prediction[idx]) == actual[idx]:
                correct_count += 1
        return correct_count / prediction_len


def test():
    with open("input.txt") as myfile:
        n, m, k = myfile.readline().split()
        sample = [next(myfile) for x in range(int(m))]
        test = [next(myfile) for x in range(int(k))]
    # print(sample)
    for i in range(len(sample)):
        # sample[i] = re.sub('\d+,', '', sample[i])
        sample[i] = sample[i].strip().split(',')
    for i in range(len(test)):
        # sample[i] = re.sub('\d+,', '', sample[i])
        test[i] = test[i].strip().split(',')
    labels = []
    for s in sample:
        labels.append(s.pop())
    attributes = [x for x in range(int(n))]
    # print('n, m, k ', n, m, k)
    print('sample ', sample)
    print('labels ', labels)
    decision_tree_model = DecisionTree(_max_depth=6, _min_splits=2)
    decision_tree_model.fit(sample, labels)
    prediction = decision_tree_model.predict(test)
    # print('test ', test)
    print(prediction, ' result')
    # print('attributes ', attributes)

    # decisionTree = DecisionTree(sample, attributes, labels)
    # print("System entropy {}".format(decisionTree.entropy))
    # decisionTree.id3()
    # decisionTree.printTree()

    test_dict = dict.fromkeys(attributes)

'''
    with open('output.txt', 'w') as f:
        for line in test:
            for i in range(len(line)):
                test_dict[attributes[i]] = line[i]
            # print(test_dict)
            # decisionTree.predict(test_dict)
            f.write(str(decisionTree.predict(test_dict)) + '\n')
            '''


def main():
    """
    Main function
    :return:
    """
    # Prepared dataset
    # use the preloaded iris dataset from sklearn for running the decision tree
    iris = load_iris()
    feature = iris.data[:, :2]
    label = iris.target
    # print(feature, ' feature')
    # print(label, ' label')
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, random_state=42)
    print(X_train, ' X_train')
    print(y_train, ' y_train')

    # Our decision tree
    decision_tree_model = DecisionTree(_max_depth=2, _min_splits=30)
    decision_tree_model.fit(X_train, y_train)
    prediction = decision_tree_model.predict(X_test)
    print(decision_tree_model.build_tree())
    # print(decision_tree_model)
    # print(prediction)

    """lets comapre preformance with sklearn decision tree based on simple accuarcy metric"""

    # decision tree from sk learn
    sk_dt_model = DecisionTreeClassifier(max_depth=2, min_samples_split=30)
    sk_dt_model.fit(X_train, y_train)
    sk_dt_prediction = sk_dt_model.predict(X_test)

    print("Our Model Accuracy : {0}".format(accuracy(prediction, y_test)))
    print(
        "SK-Learn Model Accuracy : {0}".format(accuracy(sk_dt_prediction, y_test)))

    print(list(zip(prediction, sk_dt_prediction, y_test)))


if __name__ == "__main__":
    main()
    test()
"""Output :
Our Model Accuracy : 0.7368421052631579
SK-Learn Model Accuracy : 0.7631578947368421
"""
