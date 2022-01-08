
import multiprocessing
import numpy as np

class Workers(object):

    def __init__(self, job_list, worker=None, num_workers=20):
        
        self.mgr = multiprocessing.Manager()
        self.ans = self.mgr.list()
        self.worker=worker
        self.num_workers = num_workers
        self.num_jobs = len(job_list)
        self.job_list = job_list


    def run(self, other_args=None):
        workers = [] 
        per_worker_jobs = self.num_jobs // self.num_workers + 1
        for i in range(self.num_workers):
            start = i * per_worker_jobs
            if i != self.num_workers-1: 
                end = (i + 1) * per_worker_jobs
            else:
                end = self.num_jobs
            args={
                'start':start,
                'end':end,
                'job_list':self.job_list,
                'ans':self.ans,
                'others':other_args
            }
            p = multiprocessing.Process(target=self.worker,
                                       args=(
                                           args,)
                                       )
            p.start()
            workers.append(p)
        for p in workers:
            p.join()
    
    def __call__(self):
        return self.ans
