import seaborn as sns
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
import argparse
import os
import sys
import numpy as np
import matplotlib as mpl


# width as measured in inkscape
width = 3.491
height = width / 1.618


class DatasetClass:
    def __init__(self):
        self.name = ''
        self.parentClass = -1
        self.models = []


class Evaluator:
    def __init__(self):
        self.classes = []
        self.classesQuery = []
        self.classObject = []
        self.classQuery = []
        self.distanceMatrix = None
        self.numObjects = -1
        self.numQueryObjects = -1

    def getClasses(self):
        nameClasses = [cl.name for cl in self.classes]
        return nameClasses

    def parseClassificationTarget(self, filename):
        with open(filename, 'rt') as f:
            text = f.readlines()

            firstLine = text[1].split()
            numClasses = int(firstLine[0])
            self.numObjects = int(firstLine[1])
            self.classObject = [0 for i in range(self.numObjects)]

            initIndex = 3
            for cl in range(numClasses):
                headerClass = text[initIndex].split()
                newClass = DatasetClass()
                newClass.name = headerClass[0]
                newClass.parentClass = int(headerClass[1])
                numObjectsClass = int(headerClass[2])

                initIndex = initIndex + 1

                for mod in range(numObjectsClass):
                    newClass.models.append(int(text[initIndex + mod]))
                    self.classObject[int(text[initIndex+mod])] = cl

                initIndex = initIndex + numObjectsClass

                self.classes.append(newClass)
                initIndex = initIndex + 1

    def parseClassificationQuery(self, filename):
        with open(filename, 'rt') as f:
            text = f.readlines()

            firstLine = text[1].split()
            numClasses = int(firstLine[0])
            self.numQueryObjects = int(firstLine[1])
            self.classQuery = [0 for i in range(self.numQueryObjects)]

            initIndex = 3
            for cl in range(numClasses):
                headerClass = text[initIndex].split()
                newClass = DatasetClass()
                newClass.name = headerClass[0]
                newClass.parentClass = int(headerClass[1])
                numObjectsClass = int(headerClass[2])

                initIndex = initIndex + 1

                for mod in range(numObjectsClass):
                    newClass.models.append(int(text[initIndex + mod]))
                    self.classQuery[int(text[initIndex+mod])] = cl

                initIndex = initIndex + numObjectsClass
                initIndex = initIndex + 1
                self.classesQuery.append(newClass)

    def parseDistanceMatrix(self, filename):
        self.distanceMatrix = np.loadtxt(filename)

    def computeDCG(self, result):
        dcg = []
        for i, val in enumerate(result):
            dcg.append((2**val-1)/(np.log2(i + 2)))
        return sum(dcg)

    def computeNDCG(self, result):
        perfectResult = sorted(result, reverse=True)
        return self.computeDCG(result)/self.computeDCG(perfectResult)

    def computeMetricsPerModel(self, model, value, allowedMetrics):

        # Array with precision values per recall
        recall_precision = [0 for i in range(value + 1)]

        # Init values for metrics
        NN = 0.0
        FT = 1.0
        ST = 1.0
        MAP = 0.0
        NDCG = 0.0

        # List with information about retrieval list (each element is a dictionary)
        results = []
        for i in range(self.numObjects):
            result = dict()

            # Important information to store:
            #   - the id (original position)
            #   - the object class
            #   - The distance to query

            result['id'] = i
            result['class'] = self.classObject[i]
            result['distance'] = self.distanceMatrix[model][i]
            results.append(result)

        def compareDistance(e):
            return e['distance']

        # Sort the retrieval list by distance
        results.sort(key=compareDistance)

        relevantRetrieved = 0
        numRetrieved = 0
        # Get the class of the query object
        queryClass = self.classQuery[model]
        # Get the class of the nearest neighbor object
        nearestNeighborClass = results[0]['class']
        # Get the number of target objects in the class
        numModels = len(self.classes[queryClass].models)
        rankedRelevantList = [1 if results[i]['class'] ==
                              queryClass else 0 for i in range(self.numObjects)]
        if(allowedMetrics["NDCG"]):
            NDCG = self.computeNDCG(rankedRelevantList)

        # This table stores the corresponding precision and recall in every relevant object
        precisionRecallTable = np.zeros((numModels, 2), dtype=np.float)

        while relevantRetrieved < numModels:
            if results[numRetrieved]['class'] == queryClass:
                if(allowedMetrics["NN"]):
                    if numRetrieved == 0:  # If the first retrieved is relevant, NN is 1.0
                        NN = 1.0

                # Get recall, precision values in this relevant
                rec = (relevantRetrieved + 1)/numModels
                prec = (relevantRetrieved + 1)/(numRetrieved+1)

                #
                precisionRecallTable[relevantRetrieved, 0] = rec * 100
                precisionRecallTable[relevantRetrieved, 1] = prec
                if(allowedMetrics["MAP"]):
                    MAP = MAP + prec
                relevantRetrieved = relevantRetrieved + 1

            if(allowedMetrics["FT"]):
                if numRetrieved == (numModels-1):
                    FT = (relevantRetrieved+1)/(numRetrieved+1)

            if(allowedMetrics["ST"]):
                if numRetrieved == (2*numModels-1):
                    ST = (relevantRetrieved+1)/(numRetrieved+1)

            numRetrieved = numRetrieved + 1

        MAP = MAP/numModels

        # Interpolation procedure
        index = numModels - 2
        recallValues = 100 - (100/value)
        maxim = precisionRecallTable[index+1][1]
        pos = value
        recall_precision[pos] = maxim

        pos = pos - 1

        while index >= 0:
            if int(precisionRecallTable[index][0]) >= recallValues:
                if precisionRecallTable[index][1] > maxim:
                    maxim = precisionRecallTable[index][1]
                index = index - 1
            else:
                recall_precision[pos] = maxim
                recallValues = recallValues - value
                pos = pos - 1

        while pos >= 0:
            recall_precision[pos] = maxim
            pos = pos - 1

        # The result is returned in a dictionary
        resultModel = dict()
        resultModel['pr'] = [recall_precision[i] for i in range(value+1)]
        resultModel['NN'] = NN
        resultModel['FT'] = FT
        resultModel['ST'] = ST
        resultModel['MAP'] = MAP
        resultModel['NDCG'] = NDCG
        resultModel['queryClass'] = queryClass
        resultModel['nnClass'] = nearestNeighborClass
        resultModel['rankedList'] = rankedRelevantList

        return resultModel

    def computeMetricsPerClass(self, clas, value):
        models = self.classesQuery[clas].models
        #print(f'Models : {models}')

        resultClass = dict()
        resultClass['pr'] = [0.0 for i in range(value+1)]
        resultClass['NN'] = 0.0
        resultClass['FT'] = 0.0
        resultClass['ST'] = 0.0
        resultClass['MAP'] = 0.0
        resultClass['NDCG'] = 0.0

        for model in models:
            result = self.computeMetricsPerModel(model, value)

            resultClass['pr'] = [resultClass['pr'][i] + result['pr'][i]
                                 for i in range(value + 1)]
            resultClass['NN'] = resultClass['NN'] + result['NN']
            resultClass['FT'] = resultClass['FT'] + result['FT']
            resultClass['ST'] = resultClass['ST'] + result['ST']
            resultClass['MAP'] = resultClass['MAP'] + result['MAP']
            resultClass['NDCG'] = resultClass['NDCG'] + result['NDCG']

        resultClass['pr'] = [(resultClass['pr'][i]/len(models))
                             for i in range(value + 1)]
        resultClass['NN'] = resultClass['NN']/len(models)
        resultClass['FT'] = resultClass['FT']/len(models)
        resultClass['ST'] = resultClass['ST']/len(models)
        resultClass['MAP'] = resultClass['MAP']/len(models)
        resultClass['NDCG'] = resultClass['NDCG']/len(models)

        return resultClass

    def computeMetricsAll(self, value, metrics):

        allowedMetrics = dict()
        allowedMetrics["NN"] = "NN" in metrics
        allowedMetrics["FT"] = "FT" in metrics
        allowedMetrics["ST"] = "ST" in metrics
        allowedMetrics["MAP"] = "MAP" in metrics
        allowedMetrics["NDCG"] = "NDCG" in metrics

        resultAll = dict()
        resultAll['pr'] = [0.0 for i in range(value+1)]
        resultAll['NN'] = 0.0
        resultAll['FT'] = 0.0
        resultAll['ST'] = 0.0
        resultAll['MAP'] = 0.0
        resultAll['NDCG'] = 0.0
        CM = np.zeros((len(self.classes), len(self.classes)))

        ranking = np.zeros((self.numQueryObjects, self.numObjects))
        listRanking = list()

        for i in range(self.numQueryObjects):

            result = self.computeMetricsPerModel(i, value, allowedMetrics)

            resultAll['pr'] = [resultAll['pr'][i] + result['pr'][i]
                               for i in range(value + 1)]
            resultAll['NN'] = resultAll['NN'] + result['NN']
            resultAll['FT'] = resultAll['FT'] + result['FT']
            resultAll['ST'] = resultAll['ST'] + result['ST']
            resultAll['MAP'] = resultAll['MAP'] + result['MAP']
            resultAll['NDCG'] = resultAll['NDCG'] + result['NDCG']
            CM[result['queryClass']][result['nnClass']
                                     ] = CM[result['queryClass']][result['nnClass']] + 1
            listRanking.append(result['rankedList'])

        resultAll['pr'] = [round((resultAll['pr'][i]/self.numQueryObjects) , 4)
                           for i in range(value + 1)]
        resultAll['NN'] = resultAll['NN']/self.numQueryObjects
        resultAll['FT'] = resultAll['FT']/self.numQueryObjects
        resultAll['ST'] = resultAll['ST']/self.numQueryObjects
        resultAll['MAP'] = resultAll['MAP']/self.numQueryObjects
        resultAll['NDCG'] = resultAll['NDCG']/self.numQueryObjects
        print("esto aca puede que me rompa")
        
        data = CM
        sum_per_row = data.sum(axis=1)
        CMnorm = data / sum_per_row[:, np.newaxis]
        resultAll['CM'] = np.around(CMnorm, decimals=4).tolist()
        print("esto aca no me he roto")
        cnt = 0
        for cl in self.classesQuery:
            for idx in cl.models:
                ranking[cnt] = np.asarray(listRanking[idx], dtype=np.int8)
                cnt = cnt + 1

        #resultAll['rankedList'] = ranking.tolist()

        return resultAll


class Method:
    def __init__(self):
        self.path = ''
        self.name = ''
        self.ext = []
        self.setupNames = []
        self.resultSetup = []
        self.matrices = []
        self.loadedMatrices = False

    def getNameAndSetupName(self):
        return [self.name, self.setupNames]

    def getResultSetup(self):
        return self.resultSetup

    def performEvaluation(self, evaluator, metrics, type='all', value=None, numBins=10):
        self.resultSetup.clear()

        # Load matrices for the first time
        if not self.loadedMatrices:
            for i, name in enumerate(self.setupNames):

                print(
                    self.path, self.name + self.ext[i])
                matrix = np.loadtxt(os.path.join(
                    self.path, self.name + '.' + self.ext[i]))
                print(f'{self.name} - {name}: {matrix.shape}')
                self.matrices.append(matrix)
            self.loadedMatrices = True

        for i, name in enumerate(self.setupNames):
            if type == 'all':
                evaluator.distanceMatrix = self.matrices[i]
                result = evaluator.computeMetricsAll(numBins, metrics)
                self.resultSetup.append(result)

            if type == 'class':
                evaluator.distanceMatrix = self.matrices[i]
                result = evaluator.computeMetricsPerClass(value, numBins)
                self.resultSetup.append(result)

    def selectBestPerformance(self):
        maxim = 0.0
        bestResult = None
        bestName = None

        for res, name in zip(self.resultSetup, self.setupNames):
            if res['MAP'] > maxim:
                maxim = res['MAP']
                bestResult = res
                bestName = name

        return bestResult, bestName


class Experiment:
    def __init__(self, path, outputPath, evaluator, metrics, type='all', value=None, numBins=10, reportMethods=False):

        self.styles = []
        self.metrics = metrics
        self.files = sorted(os.listdir(path))
        self.methods = dict()
        self.listMethods = []
        self.numBins = numBins
        self.evaluator = evaluator
        self.outputPath = outputPath
        self.path = path
        self.reportMethods = reportMethods
        self.type = type
        self.value = value

        for name in self.files:

            A = name.split('.')

            self.listMethods.append(".".join(A[0:-1]))

        self.listMethods = set(self.listMethods)
        self.listMethods = sorted(self.listMethods)

        for elem in self.listMethods:
            self.methods[elem] = Method()
            self.methods[elem].name = elem
            self.methods[elem].path = path

        for name in self.files:
            A = name.split('.')
            nameWithoutExt = ".".join(A[0:-1])
            self.methods[nameWithoutExt].setupNames.append("run")
            self.methods[nameWithoutExt].ext.append(A[-1])

        for k, v in self.methods.items():
            print(k)

        for key in self.methods:
            self.methods[key].performEvaluation(
                self.evaluator, metrics=self.metrics, type=type, value=value, numBins=self.numBins)

    def defineStyles(self):
        last_cycler = plt.rcParams['axes.prop_cycle']
        colors = list()
        for d in last_cycler:
            colors.append(d["color"])

        self.styles.append(dict(marker='x', color=colors[0], linestyle='--'))
        self.styles.append(dict(marker='o', color=colors[1], linestyle='-'))
        self.styles.append(dict(marker='+', color=colors[2], linestyle='-.'))
        self.styles.append(dict(marker='s', color=colors[3], linestyle=':'))
        self.styles.append(dict(marker='8', color=colors[4], linestyle='--'))
        self.styles.append(dict(marker='*', color=colors[5], linestyle='-'))
        self.styles.append(dict(marker='v', color=colors[0], linestyle='-.'))
        self.styles.append(dict(marker='p', color=colors[1], linestyle=':'))
        self.styles.append(dict(marker='D', color=colors[2], linestyle='--'))
        self.styles.append(dict(marker='.', color=colors[3], linestyle='-'))

    def generateRecallPrecisionPlotByMethod(self):
        X = np.linspace(0.0, 1.0, num=self.numBins+1)

        for (key, v) in self.methods.items():
            fig, ax = plt.subplots()
            #fig.subplots_adjust(left=.15, bottom=.2, right=0.9, top=.97)

            for i, (run, res) in enumerate(zip(v.setupNames, v.resultSetup)):
                pr = np.asarray(res['pr'])
                plt.plot(X, pr, label=run, **self.styles[i])

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.ylim(0.0, 1.0)
            plt.xlabel("Recall")
            plt.ylabel("Precision")

            fig.set_size_inches(width, height)
            fig.savefig(os.path.join(self.outputPath, v.name + '.pdf'))

    def generateRankedListByMethod(self):

        for (key, v) in self.methods.items():
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=.15, bottom=.2, right=0.9, top=.97)

            res, nameSetup = v.selectBestPerformance()
            X = res['rankedList'].astype(np.int8)
            # print(np.max(X))
            # print(X.dtype)
            #np.savetxt(v.name + '_aaa.txt', X, fmt='%d')
            plt.imshow(X, cmap='YlGn', interpolation='nearest')

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.ylim(0.0,1.0)
            # plt.xlabel("Recall")
            # plt.ylabel("Precision")

            #fig.set_size_inches(width, height)
            fig.savefig(os.path.join(self.outputPath, v.name + '_RL.pdf'))

    def generateRecallPrecisionPlot(self):
        X = np.linspace(0.0, 1.0, num=self.numBins+1)
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.15, bottom=.2, right=0.9, top=.97)
        for i, (key, v) in enumerate(self.methods.items()):
            res, nameSetup = self.methods[key].selectBestPerformance()
            res = np.asarray(res['pr'])
            plt.plot(
                X, res, label=self.methods[key].name + '('+nameSetup+')', **self.styles[i])

        #     plt.plot(X, res, label=self.methods[key].name)
        # Shrink current axis by 20%
        box = ax.get_position()
        #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.set_position([box.x0, box.y0-0.05*box.height,
                        box.width, box.height*0.9])

        # Put a legend to the right of the current axis
        if self.type == 'all':
            ax.legend(loc='upper center', bbox_to_anchor=(0.45, 1.2), ncol=4)
        #ax.legend(loc='upper center', bbox_to_anchor=(1, 0.5))
        #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0.0, 1.1)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.title(self.evaluator.classesQuery[self.value].name)
        #L = 2 * np.pi
        #x = np.linspace(0, L)
        #shift = np.linspace(0, L, 6, endpoint=False)
        # print(len(shift))
        # for i, s in enumerate(shift):
        #    ax.plot(x, np.sin(x + s), **self.styles[i])
        #plt.xlim([x[0], x[-1]])

        fig.set_size_inches(width, width)
        fig.savefig(os.path.join(self.outputPath, 'recall_precision.pdf'))

    def generateLatexTable(self):
        with open(os.path.join(self.outputPath, 'table.tex'), 'wt') as f:
            f.write('\\begin{table}\n')
            f.write('\\centering\n')
            f.write('\\begin{tabular}{| c | c | c | c | c | c |}\n')
            f.write('\\hline\n')
            f.write('Methods & NN & FT & ST & mAP & NDCG\\\\ \\hline\n')

            for i, key in enumerate(self.methods):
                name = self.methods[key].name
                for run, res in zip(self.methods[key].setupNames, self.methods[key].resultSetup):
                    f.write(
                        f'{name} ({run}) & {res["NN"]:.4f} & {res["FT"]:.4f} & {res["ST"]:.4f} & {res["MAP"]:.4f} & {res["NDCG"]:.4f} \\\\ \\hline\n')

            f.write('\\end{tabular}\n')
            f.write('\\caption{Table}\n')
            f.write('\\end{table}\n')

    def generateTextTable(self):
        with open(os.path.join(self.outputPath, 'table.txt'), 'wt') as f:
            f.write('Methods \t NN \t FT \t ST \t mAP \t NDCG\n')

            for i, key in enumerate(self.methods):
                name = self.methods[key].name
                for run, res in zip(self.methods[key].setupNames, self.methods[key].resultSetup):
                    f.write(
                        f'{name} ({run}) \t {res["NN"]:.3f} \t {res["FT"]:.3f} \t {res["ST"]:.3f} \t {res["MAP"]:.3f} \t {res["NDCG"]:.3f} \n')

    def plotConfussionMatrix(self, name, nameSetup, matrix):
        nameClasses = [cl.name for cl in self.evaluator.classes]

        data = matrix
        sum_per_row = data.sum(axis=1)
        dataNorm = data / sum_per_row[:, np.newaxis]

        f, ax = plt.subplots()
        f.subplots_adjust(left=.25, bottom=.25)
        heatmap = sns.heatmap(dataNorm, cbar=False, cmap='viridis', vmin=0.0, vmax=1.0,
                              square=True, linewidths=.5, xticklabels=nameClasses, yticklabels=nameClasses)
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=90, fontsize=14)
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(), rotation=0, fontsize=14)
        plt.title(name + '('+nameSetup+')')
        f.set_size_inches(width, width)
        f.savefig(os.path.join(self.outputPath, f'cm_{name}_{nameSetup}.pdf'))

        plt.close('all')

    def generateConfussionMatrices(self):
        for i, (key, v) in enumerate(self.methods.items()):
            name = v.name

            if not self.reportMethods:  # Plot the best configuration only
                res, nameSetup = v.selectBestPerformance()
                self.plotConfussionMatrix(name, nameSetup, res['CM'])
            else:  # Plot all the configurations
                for run, res in zip(v.setupNames, v.resultSetup):
                    print(f'{name} - {run}')
                    print(res['CM'])
                    self.plotConfussionMatrix(name, run, res['CM'])

    def makeCombinationMatrices(self):
        L = list()
        for i, (key, v) in enumerate(self.methods.items()):
            name = v.name
            res, nameSetup = v.selectBestPerformance()
            D = np.loadtxt(os.path.join(self.path, f'{name}.{nameSetup}.txt'))
            L.append((name, nameSetup, D/np.max(D)))

        for i in range(len(L)-1):
            for j in range(i+1, len(L)):
                newD = L[i][2] + L[j][2]
                np.savetxt(L[i][0] + '+' + L[j][0] + '.run1.txt', newD)

    def getMethods(self):
        # Entrega el primer key (siempre debería haber una sola)
        res = list(self.methods.keys())[0]

        # entrega el primer resultSetup (siempre debería haber uno solo)
        my_dict = self.methods[res]
        print(my_dict)

        return my_dict
