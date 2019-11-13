# пробный сплит по атрибуту и его значению
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# Вычисление Индекса Джини для сплита
def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Выбор лучшей точки сплита основываясь на индексе Джини
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    # print(class_values, ' class_values')
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Функция создания терминального узла
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Проверка на создание дочерних сплитов узла либо создание терминального узла
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

"""
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))
		"""


# Сделать предсказание
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


def main():
    with open("data_banknote_authentication.txt") as myfile:
        n, m, k = myfile.readline().split()
        sample = [next(myfile) for x in range(int(m))]
        test = [next(myfile) for x in range(int(k))]
    # print(sample)

    for i in range(len(sample)):
        sample[i] = sample[i].strip().split(',')

    for i in range(len(test)):
        test[i] = test[i].strip().split(',')
    # print('n, m, k ', n, m, k)
    # print('sample ', sample)
    # print('test ', test)

    tree = build_tree(sample, int(n)-3, 1)

    with open('output.txt', 'w') as f:
        for line in test:
            prediction = predict(tree, line)
            f.write(str(prediction + '\n'))
'''
    for line in test:
        prediction = predict(tree, line)
        print(prediction)

    # Проверка accuracy предсказаний
    for row in sample:
        prediction = predict(tree, row)
        print('Expected={0}, Got={1}'.format(row[-1], prediction))
    '''


if __name__ == '__main__':
    main()
