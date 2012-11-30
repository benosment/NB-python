#! /usr/bin/env python
#
# Ben Osment
# Sun Nov 11 15:09:47 2012

import sys
import math
import pdb
import random

class TrainingExample:
    def __init__(self, class_label, attributes):
        self.class_label = class_label
        self.attributes = attributes


class NaiveBayes:
    def __init__(self, attribute_values):
        # initial count of attributes
        #self.N = 0
        # list of training data points
        self.training_data = []
        self.N = 0 # total number of training instances
        self.pos_cnt = 0 # positive training instances
        self.neg_cnt = 0 # negative training instances
        self.pos_freq = []
        self.neg_freq = []
        self.attribute_values = attribute_values

    def train(self, train_data):
        # read a file, add each line as a new example
        created_freq = False
        #print "train: ", len(train_data), len(train_data[0])
        for di in train_data:
            class_label = di[0]
            data_instance = di[1:]

            if not created_freq:
                created_freq = True
                # create an array of dictionaries to store the
                # frequency of each attribute
                for attribute in data_instance:
                    self.pos_freq.append({})
                    self.neg_freq.append({})
                    #self.pos_freq = [{}] * len(attributes)
                    #self.neg_freq = [{}] * len(attributes)
            if class_label == '+1':
                self.pos_cnt += 1
                for i in range(len(data_instance)):
                    self.pos_freq[i][data_instance[i]] = self.pos_freq[i].get(data_instance[i], 0) + 1
            else:
                self.neg_cnt += 1
                for i in range(len(data_instance)):
                    self.neg_freq[i][data_instance[i]] = self.neg_freq[i].get(data_instance[i], 0) + 1
            self.N += 1
            #example = TrainingExample(class_label, attribute)
        #print "n=%d, num_pos=%d, num_neg=%d num_features=%d" % \
        #(self.N, self.pos_cnt, self.neg_cnt, len(data_instance))

        # create probability from frequency counts
        self.p_pos = math.log(self.pos_cnt / float(self.N))
        self.p_neg = math.log(self.neg_cnt / float(self.N))
        self.p_pos_attr = []
        self.p_neg_attr = []
        for i in range(len(self.pos_freq)):
            d = {}
            denom = float(self.pos_cnt) + len(self.attribute_values[i])
            for key in self.attribute_values[i]:
                num = 1 + self.pos_freq[i].get(key,0) #laplace
                d[key] = math.log(num / float(denom))
            self.p_pos_attr.append(d)
        for i in range(len(self.neg_freq)):
            d = {}
            denom = float(self.neg_cnt) + len(self.attribute_values[i])
            for key in self.attribute_values[i]:
                num = 1 + self.neg_freq[i].get(key,0) #laplace
                d[key] = math.log(num / float(denom))
            self.p_neg_attr.append(d)

    def test(self, data):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        errors = []
        for di in data:
            class_label = di[0]
            attributes = di[1:]
            predicted_label = self.classify(attributes)
            if class_label == '+1':
                if predicted_label == '+1':
                    tp += 1
                    errors.append(0)
                else:
                    fn += 1
                    errors.append(-1)
            else:
                if predicted_label == '-1':
                    tn += 1
                    errors.append(0)
                else:
                    fp += 1
                    errors.append(-1)
        return tp, tn, fp, fn, errors

    def classify(self, attribute_vector):
        pos_prob = self.p_pos
        neg_prob = self.p_neg
#        try:
        for i in range(len(attribute_vector)):
            pos_prob += self.p_pos_attr[i][attribute_vector[i]]
            neg_prob += self.p_neg_attr[i][attribute_vector[i]]
#        except:
#            pdb.set_trace()
        #print pos_prob, neg_prob
        if pos_prob > neg_prob:
            return '+1'
        else:
            return '-1'

def read_data(filename):
    labels = []
    data = []
    full = [] 
    for line in open(filename):
        data_instance = line.strip().split('\t')
        full.append(data_instance[:])
        class_label = data_instance.pop(0)
        labels.append(class_label)
        data.append(data_instance)
    return full, labels, data


def test_ensemble(classifiers, filename):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for line in open(filename):
        data_instance = line.strip().split('\t')
        class_label = data_instance.pop(0)
        prediction = 0.0
        for (classifier, alpha) in classifiers:
            predicted_label = classifier.classify(data_instance)
            if predicted_label == '+1':
                prediction += 1.0 * alpha
            else:
                prediction -= 1.0 * alpha

        if class_label == '+1':
            if prediction > 0:
                tp += 1
            else:
                fn += 1
        else:
            if prediction < 0:
                tn += 1
            else:
                fp += 1

    return tp, tn, fp, fn
            
class WeightedRandomSample:
    def __init__(self, data, weights):
        self.data = data
        weight_sum = 0.0
        self.weights = []
        for weight in weights:
            weight_sum += weight
            self.weights.append(weight_sum)

    def sample(self):
        # take a random sample
        random_weight = random.random()
        # find which element this corresponds to
        i = 0
        while (random_weight > self.weights[i]):
            i += 1
        return self.data[i]

if __name__ == '__main__':
    # TODO: add num iterations
    if len(sys.argv) != 3:
        print "Usage: nb.py training_file testing_file"
        sys.exit()

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]

    train_full, train_labels, train_data  = read_data(train_filename)

    created_freq = False
    attribute_values = []
    # get set of all attributes
    for data_instance in train_data:
        if not created_freq:
            created_freq = True
            # create an array of dictionaries to store the
            # frequency of each attribute
            for attribute in data_instance:
                attribute_values.append([])
        for i in range(len(data_instance)):
            attribute_values[i].append(data_instance[i])

    # create a set out of each of the attributes
    for i in range(len(attribute_values)):
        attribute_values[i] = set(attribute_values[i])

    #print len(train_set)

    # need to account for label
    num_train = len(train_data)
    # initial have D (weights) set to 1/|D|
    D = [1.0/len(train_data) for data in train_data]

    classifier_list = []

    for i in range(8):
        nb = NaiveBayes(attribute_values[:])
        # create a weighted random sample based off of D
        wrs = WeightedRandomSample(train_full, D)

        # sample N times
        random_sample = [wrs.sample() for i in range(num_train)]
        # train classifier using that sample
        nb.train(random_sample)
        # classify exisiting training set
        # TODO: use the full set...right? 
        # test accuracy by classifying the (original) training  set
        tp, tn, fp, fn, errors = nb.test(train_full) 
        #print tp, tn, fp, fn
        accuracy = float((tp+tn)) / (tp+tn+fp+fn)

        # calculate alpha
        alpha = 0.5 * math.log(accuracy / max((1.0-accuracy), 1e-16))
        #print "alpha=", alpha, "accuracy=", accuracy

        # adjust weights
        for j in range(len(errors)):
            if errors[j] == -1:
                # prediction was incorrect, increase weight
                #print "wrong increasing weight old D" , D[j], "alpha", alpha
                D[j] = D[j] * math.exp(alpha)
                #print D[j]
            else:
                # prediction was correct, decrease weight
                #print "right decreasing weight old D" , D[j], "alpha", alpha
                D[j] = D[j] * math.exp(-alpha)
                #print D[j]
        # normalize D so that is sums to 1
        D_sum = sum(D)
        D = [D_weight/D_sum for D_weight in D]

        classifier_list.append((nb, alpha))

        if (fp == fn == 0):
            break

    # evaluate performance
    tp, tn, fp, fn = test_ensemble(classifier_list, test_filename)

    # test_full, test_labels, test_data  = read_data(test_filename)
    # tp, tn, fp, fn = nb.test(test_full)
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    print "True Positive:", tp
    print "True Negative:", tn
    print "False Positive:", fp
    print "False Negative:", fn
    print "Accuracy:", (tp+tn) / (tp+tn+fp+fn)
    print "Error Rate:", (fp+fn) / (tp+tn+fp+fn)
    print "Sensitivity:", tp/(tp+fn)
    print "Specificity:", tn/(fp+tn)
    print "Precision:", tp/(tp+fp)
    
