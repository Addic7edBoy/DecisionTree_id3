import re
import math
from collections import deque


class Node(object):
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTree(object):
    def __init__(self, sample, attributes, labels):
        self.sample = sample
        self.attributes = attributes
        self.labels = labels
        self.labelCodes = None
        self.labelCodesCount = None
        self.initLabelCodes()
        # print(self.labelCodes, ' labelCodes')
        # print(self.labelCodesCount, 'labelCodesCount')
        self.root = None
        self.entropy = self.getEntropy([x for x in range(len(self.labels))])

    def initLabelCodes(self):
        self.labelCodes = []
        self.labelCodesCount = []
        for l in self.labels:
            if l not in self.labelCodes:
                self.labelCodes.append(l)
                self.labelCodesCount.append(0)
            self.labelCodesCount[self.labelCodes.index(l)] += 1

    def getLabelCodeId(self, sampleId):
        return self.labelCodes.index(self.labels[sampleId])

    def getAttributeValues(self, sampleIds, attributeId):
        vals = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in vals:
                vals.append(val)
        return vals

    def getEntropy(self, sampleIds):
        entropy = 0
        labelCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCount[self.getLabelCodeId(sid)] += 1
        for lv in labelCount:
            if lv != 0:
                entropy += -lv / len(sampleIds) * \
                    math.log(lv / len(sampleIds), 2)
            else:
                entropy += 0
        return entropy

    def getDominantLabel(self, sampleIds):
        labelCodesCount = [0] * len(self.labelCodes)
        for sid in sampleIds:
            labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
        return self.labelCodes[labelCodesCount.index(max(labelCodesCount))]

    def getInformationGain(self, sampleIds, attributeId):
        gain = self.getEntropy(sampleIds)
        attributeVals = []
        attributeValsCount = []
        attributeValsIds = []
        for sid in sampleIds:
            val = self.sample[sid][attributeId]
            if val not in attributeVals:
                attributeVals.append(val)
                attributeValsCount.append(0)
                attributeValsIds.append([])
            vid = attributeVals.index(val)
            attributeValsCount[vid] += 1
            attributeValsIds[vid].append(sid)
        # print("-gig", self.attributes[attributeId])
        for vc, vids in zip(attributeValsCount, attributeValsIds):
            # print("-gig", vids)
            gain -= vc / len(sampleIds) * self.getEntropy(vids)
        return gain

    def getAttributeMaxInformationGain(self, sampleIds, attributeIds):
        attributesEntropy = [0] * len(attributeIds)
        for i, attId in zip(range(len(attributeIds)), attributeIds):
            attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
        maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
        return self.attributes[maxId], maxId

    def isSingleLabeled(self, sampleIds):
        label = self.labels[sampleIds[0]]
        for sid in sampleIds:
            if self.labels[sid] != label:
                return False
        return True

    def getLabel(self, sampleId):
        return self.labels[sampleId]

    def id3(self):
        sampleIds = [x for x in range(len(self.sample))]
        attributeIds = [x for x in range(len(self.attributes))]
        # print(sampleIds, attributeIds)
        self.root = self.id3Recv(sampleIds, attributeIds, self.root)

    def id3Recv(self, sampleIds, attributeIds, root):
        root = Node()  # Initialize current root
        if self.isSingleLabeled(sampleIds):
            root.value = self.labels[sampleIds[0]]
            return root
        # print(attributeIds)
        if len(attributeIds) == 0:
            root.value = self.getDominantLabel(sampleIds)
            return root
        bestAttrName, bestAttrId = self.getAttributeMaxInformationGain(
            sampleIds, attributeIds)
        # print(bestAttrName)
        root.value = bestAttrName
        root.childs = []  # Create list of children
        for value in self.getAttributeValues(sampleIds, bestAttrId):
            # print(value)
            child = Node()
            child.value = value
            root.childs.append(child)  # Append new child node to current
            childSampleIds = []
            for sid in sampleIds:
                if self.sample[sid][bestAttrId] == value:
                    childSampleIds.append(sid)
            if len(childSampleIds) == 0:
                child.next = self.getDominantLabel(sampleIds)
            else:
                # print(bestAttrName, bestAttrId)
                # print(attributeIds)
                if len(attributeIds) > 0 and bestAttrId in attributeIds:
                    toRemove = attributeIds.index(bestAttrId)
                    attributeIds.pop(toRemove)
                child.next = self.id3Recv(
                    childSampleIds, attributeIds, child.next)
        return root

    def printTree(self):
        if self.root:
            roots = deque()
            roots.append(self.root)
            # print('deque ', str(roots))
            while len(roots) > 0:
                root = roots.popleft()
                print(root.value)
                if root.childs:
                    for child in root.childs:
                        print('({})'.format(child.value))
                        roots.append(child.next)
                elif root.next:
                    print(root.next)

    def predict(self, test_dict):
        if self.root:
            roots = deque()
            roots.append(self.root)
            # print({k:list(v) for k,v in roots.items})
            # print('deque ', str(roots))
            while len(roots) > 0:
                root = roots.popleft()
                if root.value in test_dict.keys():
                    # print(root.value)
                    x = test_dict.pop(root.value)
                if root.childs:
                    for child in root.childs:
                        if child.value == x:
                            # print('({})'.format(child.value))
                            roots.append(child.next)
                            break
                else:
                    # print(root.value)
                    return root.value

'''
def test():
    f = open('input.txt')
    attributes = f.readline().split(',')
    attributes = attributes[1:len(attributes) - 1]
    sample = f.readlines()
    f.close()
    for i in range(len(sample)):
        sample[i] = re.sub('\d+,', '', sample[i])
        sample[i] = sample[i].strip().split(',')
    labels = []
    for s in sample:
        labels.append(s.pop())
    print(labels)
    decisionTree = DecisionTree(sample, attributes, labels)
    print("System entropy {}".format(decisionTree.entropy))
    decisionTree.id3()
    decisionTree.printTree()

    predict_attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    predict_samples = ["Sunny", 'Cool', 'Normal', 'Weak']
    predict_dict = dict(Outlook='Sunny', Temperature='Cool',
                        Humidity='Normal', Wind='Weak')
    decisionTree.predict(predict_attributes, predict_dict)
'''

def main():
    with open("input1.txt") as myfile:
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
    # rint('sample ', sample)
    # print('labels ', labels)
    # print('test ', test)
    # print('attributes ', attributes)

    decisionTree = DecisionTree(sample, attributes, labels)
    # print("System entropy {}".format(decisionTree.entropy))
    decisionTree.id3()
    decisionTree.printTree()

    test_dict = dict.fromkeys(attributes)

    with open('output.txt', 'w') as f:
        for line in test:
            for i in range(len(line)):
                test_dict[attributes[i]] = line[i]
            # print(test_dict)
            # decisionTree.predict(test_dict)
            f.write(str(decisionTree.predict(test_dict)) + '\n')

if __name__ == '__main__':
    main()
