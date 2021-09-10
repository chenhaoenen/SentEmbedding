# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/9/10 9:36 
# Description:  
# --------------------------------------------
import sys
PATH_ENTEVAL = './SentEval'
PATH_SENTEVAL_DATA = './SentEval/data'
sys.path.insert(0, PATH_ENTEVAL)
import senteval

def evaluate(batch_func, eval_senteval_transfer=False):

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        return batch_func(batch)

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_SENTEVAL_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    if eval_senteval_transfer:
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    results = se.eval(tasks)

    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman,
               "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}
    if eval_senteval_transfer:
        avg_transfer = 0
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            avg_transfer += results[task]['devacc']
            metrics['eval_{}'.format(task)] = results[task]['devacc']
        avg_transfer /= 7
        metrics['eval_avg_transfer'] = avg_transfer

    return metrics


