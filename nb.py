#! /usr/bin/env python
#
# Ben Osment
# Sun Nov 11 15:09:47 2012

import sys
import math
import pdb

class TrainingExample:
    def __init__(self, class_label, attributes):
        self.class_label = class_label
        self.attributes = attributes


class NaiveBayes:
    def __init__(self):
        # initial count of attributes
        # list of training data points
        self.training_data = []
        self.N = 0 # total number of training instances
        self.pos_cnt = 0 # positive training instances
        self.neg_cnt = 0 # negative training instances
        self.pos_freq = []
        self.neg_freq = []
        self.attribute_values = []

    def train(self, train_filename):
        # read a file, add each line as a new example
        created_freq = False
        for line in open(train_filename):
            data_instance = line.strip().split('\t')
            class_label = data_instance.pop(0)
            attributes = data_instance
            if not created_freq:
                created_freq = True
                # create an array of dictionaries to store the
                # frequency of each attribute
                for attribute in attributes:
                    self.pos_freq.append({})
                    self.neg_freq.append({})
                    self.attribute_values.append([])
            if class_label == '+1':
                self.pos_cnt += 1
                for i in range(len(attributes)):
                    self.attribute_values[i].append(attributes[i])
                    self.pos_freq[i][attributes[i]] = self.pos_freq[i].get(attributes[i], 0) + 1
            else:
                self.neg_cnt += 1
                for i in range(len(attributes)):
                    self.attribute_values[i].append(attributes[i])
                    self.neg_freq[i][attributes[i]] = self.neg_freq[i].get(attributes[i], 0) + 1
            self.N += 1
        print "n=%d, num_pos=%d, num_neg=%d num_features=%d" % \
        (self.N, self.pos_cnt, self.neg_cnt, len(attributes))
        # create probability from frequency counts
        self.p_pos = math.log(self.pos_cnt / float(self.N))
        self.p_neg = math.log(self.neg_cnt / float(self.N))
        self.p_pos_attr = []
        self.p_neg_attr = []
        # create a set out of each of the attributes
        for i in range(len(self.attribute_values)):
            self.attribute_values[i] = set(self.attribute_values[i])
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

        #print "attribute features", self.attribute_values
        #print "p_pos_class", self.p_pos
        #print "p_neg_class", self.p_neg
        #print
        #print "pos_p", self.p_pos_attr
        #print
        #print "neg_p", self.p_neg_attr
#        pdb.set_trace()
            
    def test(self, test_filename):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for line in open(test_filename):
            data_instance = line.strip().split('\t')
            class_label = data_instance.pop(0)
            #pdb.set_trace()
            predicted_label = self.classify(data_instance)
            if class_label == '+1':
                if predicted_label == '+1':
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted_label == '-1':
                    tn += 1
                else:
                    fp += 1
        return tp, tn, fp, fn

    def classify(self, attribute_vector):
        pos_prob = self.p_pos
        neg_prob = self.p_neg
        for i in range(len(attribute_vector)):
            pos_prob += self.p_pos_attr[i][attribute_vector[i]]
            neg_prob += self.p_neg_attr[i][attribute_vector[i]]

        #print pos_prob, neg_prob
        if pos_prob > neg_prob:
            return '+1'
        else:
            return '-1'

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage: nb.py training_file testing_file"
        sys.exit()
    nb = NaiveBayes()
    # parse the training data
    nb.train(sys.argv[1])
    # evaluate performance
    tp, tn, fp, fn = nb.test(sys.argv[2])
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
    
