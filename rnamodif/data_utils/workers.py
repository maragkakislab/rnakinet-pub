import torch
import time

def worker_init_fn_multisplit(worker_id):
    #TODO new split structute with 'exp' -> adjust workers? or datamodule?
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
        
    dataset.positive_files = slice_splits(dataset.positive_files, total_workers, current_worker)
    dataset.negative_files = slice_splits(dataset.negative_files, total_workers, current_worker)
    
    
def slice_splits(split, total_workers, current_worker):
    new_split = []
    for exp, files in split:
        new_files = get_files_partition(files, total_workers, current_worker)
        new_split.append((exp,new_files))
    return new_split

def get_files_partition(files, total, current):
    files_per_worker = len(files)//total
    
    begin_index = files_per_worker*current
    end_index = files_per_worker*(current+1)
    if(current == total-1): #Last worker
        result = files[begin_index:]
    else:
        result = files[begin_index:end_index]
    assert(len(result)>0)
    return result
    

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
    
    # time.sleep(2)
    # print('TOTAL_WORKERS', total_workers)
    # print('CURRENT WORKER', current_worker)
    # time.sleep(1)
    
    pos_per_worker = len(dataset.positive_files)//total_workers
    neg_per_worker = len(dataset.negative_files)//total_workers
    
    if(current_worker == total_workers -1): #Last worker
        # print('LAST WORKER')
        dataset.positive_files = dataset.positive_files[pos_per_worker*current_worker:]
        dataset.negative_files = dataset.negative_files[neg_per_worker*current_worker:]
    else:
        dataset.positive_files = dataset.positive_files[pos_per_worker*current_worker: pos_per_worker*(current_worker+1)]
        dataset.negative_files = dataset.negative_files[neg_per_worker*current_worker: neg_per_worker*(current_worker+1)] 
       
    # print(sorted([int(Path(x).stem.split('_')[-1]) for x in dataset.positive_files]))
    # print(sorted([int(Path(x).stem.split('_')[-1]) for x in dataset.negative_files]))
    
    assert(len(dataset.positive_files)>0)
    assert(len(dataset.negative_files)>0)
    

def worker_init_simple_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
    files_per_worker = len(dataset.files)//total_workers
    
    #TODO inefficient split, can result in workers = 16 , split = 1,1,1,1...15
    if(current_worker == total_workers -1): #Last worker
        # print('LAST WORKER')
        dataset.files = dataset.files[files_per_worker*current_worker:]
    else:
        dataset.files = dataset.files[files_per_worker*current_worker: files_per_worker*(current_worker+1)]
       
    # print(sorted([int(Path(x).stem.split('_')[-1]) for x in dataset.positive_files]))
    
    assert(len(dataset.files)>0)

def worker_init_event_batch_fn_single_dset(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    total_workers = worker_info.num_workers
    current_worker = worker_id
    
    pos_b = dataset.pos_dset.available_batch_names
    neg_b = dataset.neg_dset.available_batch_names
    
    if(total_workers <= len(pos_b)):
        dataset.pos_dset.available_batch_names = [el for i,el in enumerate(pos_b) if i%total_workers == current_worker]
    else:
        dataset.pos_dset.available_batch_names = [el for i,el in enumerate(pos_b) if current_worker%len(pos_b) == i]
       
    if(total_workers <= len(neg_b)):
        dataset.neg_dset.available_batch_names = [el for i,el in enumerate(neg_b) if i%total_workers == current_worker]
    else: 
        dataset.neg_dset.available_batch_names = [el for i,el in enumerate(neg_b) if current_worker%len(neg_b) == i]
    
    # print('worker', current_worker)
    # print('pos batches', dataset.pos_dset.available_batch_names)
    # print('neg_batches', dataset.neg_dset.available_batch_names)
       
    assert(len(dataset.pos_dset.available_batch_names)>0)
    assert(len(dataset.neg_dset.available_batch_names)>0) 
    
    
# def worker_init_event_batch_fn(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     total_workers = worker_info.num_workers
#     current_worker = worker_id
    
#     for dset in dataset.pos_dsets+dataset.neg_dsets:
#         print(dset.available_batch_names)
    
#     for dset in dataset.pos_dsets+dataset.neg_dsets:
#         b = dset.available_batch_names
    
#         if(total_workers <= len(b)):
#             b = [el for i,el in enumerate(b) if i%total_workers == current_worker]
#         else:
#             b = [el for i,el in enumerate(b) if current_worker%len(b) == i]
    
#         assert(len(b)>0)
    
#     print('after')
#     for dset in dataset.pos_dsets+dataset.neg_dsets:
#         print(dset.available_batch_names)