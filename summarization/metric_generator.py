import pandas as pd

from util.metrics import *


class MetricGenerator():
    def __init__(self, metrics=None):
        if metrics is None:
            metrics = ['precision', 'recall', 'f1-score', 'accuracy']
        self.metrics = {}
        self.counts = 0
        for i in metrics:
            self.metrics[i] = 0

    def update(self, output, target):
        counts = len(target)
        for metric in self.metrics.keys():
            ''' compute metric '''
            if metric == 'precision':
                score = get_precision_score(output, target)
            elif metric == 'recall':
                score = get_recall_score(output, target)
            elif metric == 'f1-score':
                score = get_f1_score(output, target)
            elif metric == 'accuracy':
                score = get_accuracy_score(output, target)
            else:
                continue
            #print(score)
            ''' update '''
            num_a = self.counts
            num_b = counts
            num = num_a + num_b
            ratio_a = num_a / num if num > 0 else 0
            ratio_b = num_b / num if num > 0 else 0
            self.metrics[metric] = self.metrics[metric] * ratio_a + score * ratio_b
        self.counts += counts

    def get_metrics(self):
        return self.metrics

    def save_to_csv(self, path, file_name):
        df = pd.DataFrame(self.metrics.items(), columns=list(self.metrics.keys()))
        # Write DataFrame to csv file
        df.to_csv(f'{path}/{file_name}.csv')

    def save_to_excel(self, path, file_name):
        df = pd.DataFrame(self.metrics.items(), columns=list(self.metrics.keys()))
        # Write DataFrame to Excel file
        df.to_excel(f'{path}/{file_name}.xlsx')


if __name__ == '__main__':
    mg = MetricGenerator(num_classes=2)
    preds = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    tures = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mg.update(preds, tures)
