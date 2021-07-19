# -*- coding: UTF-8 -*-
import numpy
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
class TrAdaBoost():
    def __int__(self):
        pass

    def tradaboost(self, wts, clf, Source_data, Target_data, Source_label, Target_label, Test_data, Test_label, N):
        # X_train for trans_S, X_test for test, trans_T for trans_A
        trans_data = np.concatenate((Target_data, Source_data), axis=0)
        trans_label = np.concatenate((Target_label, Source_label), axis=0)

        row_A = Target_data.shape[0]  # Here we only need the weights from row_A aka trans_T aka public dataset
        row_S = Source_data.shape[0]
        row_T = Test_data.shape[0]

        test_data = np.concatenate((trans_data, Test_data), axis=0)

        # 初始化权重
        weights_A = np.ones(row_A) / row_A
        weights_S = np.ones(row_S) / row_S
        weights = np.concatenate((weights_A, weights_S), axis=0)

        beta = 0.5 * np.log(1 + np.sqrt(2 * np.log(row_A / N)))  # use setting from MultiSourceTrAdaboost

        # Store beta_T & result labels
        beta_T = np.zeros([1, N])
        result_label = np.zeros([row_A + row_S + row_T, N])  # zeros?

        predict = np.zeros([row_T])

        print('params initial finished.')
        trans_data = np.asarray(trans_data, order='C')
        trans_label = np.asarray(trans_label, order='C')
        test_data = np.asarray(test_data, order='C')

        for i in range(N):
            P = self.calculate_P(weights, trans_label)

            result_label[:, i], updated_model = self.train_classify(clf, trans_data, trans_label,
                                                                    test_data, P)

            print("Accuracy score for iteration {i}: {acc}".format(i=i,
                                                                   acc=accuracy_score(result_label[row_A + row_S:, i],
                                                                                      Test_label)))
            print('Results:', result_label[row_A + row_S:, i])

            # error_rate = self.calculate_error_rate(Source_label, result_label[row_A:row_A + row_S, i],
            #                                        weights[row_A:])  # error_rate from source, [incorrect]
            # print('public data weight sum: ', np.sum(weights[row_A:]))
            error_rate = self.calculate_error_rate(Target_label, result_label[:row_A, i],
                                                   weights[:row_A])  # error_rate from target [correct but break soon]

            # error_rate = 1 - accuracy_score(result_label[:row_A, i], Target_label, sample_weight = weights[:row_A])  # no weight for beta & error

            print('Error rate for iteration {i}: {rate}'.format(i=i, rate=error_rate))
            if error_rate > 0.5:
                error_rate = 0.49
            if error_rate == 0:
                N = i
                break  # Prevent over-fit

            # beta_T[0, i] = error_rate / (1 - error_rate)  # Original way from TrAdaBoost
            beta_T[0, i] = 0.25 * np.log((1 - error_rate) / error_rate)  # Improved by MultiSourceTrAdaBoost
            print('Beta_T: ', beta_T[0, i])

            # Adjust target sample weights
            label = np.zeros(row_S)
            for j in range(row_S):
                if result_label[row_A + j, i] != Source_label[j]:
                    label[j] = 1
                else:
                    label[j] = 0
            for j in range(row_S):
                # weights[row_A + j] = weights[row_A + j] * np.power(beta_T[0, i], -label[j])
                weights[row_A + j] = weights[row_A + j] * np.exp(-label[j] * beta)

            # Adjust source sample weights
            label = np.zeros(row_A)
            for j in range(row_A):
                if result_label[j, i] != Target_label[j]:
                    label[j] = 1
                else:
                    label[j] = 0
            for j in range(row_A):
                # weights[j] = weights[j] * np.power(beta, label[j])
                weights[j] = weights[j] * np.exp(label[j] * beta_T[0, i])

            # Weights
            # for j in range(row_S):
            #     if result_label[row_A + j, i] != Source_label[j]:
            #         weights[row_A + j] = weights[row_A + j] * beta_T[0, i]
            # for j in range(row_A):
            #     if result_label[j, i] != Target_label[j]:
            #         weights[j] = weights[j] / beta

        print('Beta_T: ', beta_T)  # Previous logic: beta_T low = good, now: beta_T high = good, all weights need change
        # print beta_T
        count_label = np.zeros([row_T])
        for i in range(row_T):
            # calculate left, right. In original algorithm, use left & right to return predict 0 & 1.
            left = np.sum(
                result_label[row_A + row_S + i, 0:int(np.ceil(N / 4))] * np.log(1 / beta_T[0, 0:int(np.ceil(N / 4))]))
            right = 0.5 * np.sum(np.log(1 / beta_T[0, 0:int(np.ceil(N / 4))]))

            #  Here is the prediction process, this is a binary labels, we should try it with multiple labels
            count_label[i] = left - right
            # print(count_label[i])  # Check different labels
        count_max = count_label.max()
        count_min = count_label.min()
        count_label = (count_label - count_min) / (count_max - count_min)

        for i in range(row_T):  # label = [0,1,2,3,4,5]
            if count_label[i] <= 0.106:
                predict[i] = 0
            elif count_label[i] <= 0.261:
                predict[i] = 1
            elif count_label[i] <= 0.471:
                predict[i] = 2
            elif count_label[i] <= 0.715:
                predict[i] = 3
            elif count_label[i] <= 0.91:
                predict[i] = 4
            else:
                predict[i] = 5

        Public_wts = weights[row_A:]  # * row_A

        return predict, updated_model, Public_wts.T

    def calculate_P(self, weights, label):
        # for i in range(len(weights)):
        #     if weights[i] > 1.0:
        #         weights[i] = 1.0
        #     else:
        #         continue
        total = np.sum(weights)
        return np.asarray(weights / total, order='C')

    def train_classify(self, clf, trans_data, trans_label, test_data, P):  # AdaBoost
        clf.fit(trans_data, trans_label, sample_weight=P[0:len(trans_label), ])
        updated_model = clf
        # for clf, w in zip(clf.estimators_, clf.estimator_weights_):
        #     updated_model.append(clf.coef_.reshape(-1)*w)
        # updated_model = np.array(updated_model).mean(axis=0)
        return clf.predict(test_data), updated_model

    def calculate_error_rate(self, label_R, label_H, weight):
        total = np.sum(weight)
        label = numpy.ones(label_R.shape[0])
        for i in range(len(label_R)):
            if label_H[i] != label_R[i]:
                label[i] = 1
            else:
                label[i] = 0
        print(np.sum(label))
        # print(total)
        # return np.sum(weight * label) / total  # 5 labels
        return np.sum(weight * label / total)     # From MultiSourceTrAdaBoost
        # total = np.sum(weight)
        # error = np.sum(weight * np.abs(label_R - label_H) / total)
        # return error # np.sum(weight[:, 0] / total * np.abs(label_R - label_H))
