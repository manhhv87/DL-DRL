import torch
from torch.utils.data import DataLoader
from utils import move_to
from tqdm import tqdm
import random
import os
from utils.data_utils import save_dataset, load_dataset

def Lalternate_val_dataset(model, Lmodel, val_dataset, opts):

    print('Generating lower-level validation data')
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
        model.load_state_dict({**model.state_dict(), **{}.get('model', {})})
    if isinstance(Lmodel,torch.nn.DataParallel):
        Lmodel = Lmodel.module
        Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get('model', {})})
    model.eval()
    model.cuda()
    model.set_decode_type('greedy')
    Val_dataset = DataLoader(val_dataset, batch_size=opts.eval_batch_size, num_workers=0)
    val_datasets = []
    with torch.no_grad():
        for i in tqdm(Val_dataset, disable=opts.no_progress_bar):
            Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4, Lmask_dataset5, Lmask_dataset6 = model(move_to(i, opts.device), Lmodel, Lval_dataset=True)
            val_datasets.append(Lmask_dataset1)
            val_datasets.append(Lmask_dataset2)
            val_datasets.append(Lmask_dataset3)
            val_datasets.append(Lmask_dataset4)
            val_datasets.append(Lmask_dataset5)
            val_datasets.append(Lmask_dataset6)

    torch.cuda.empty_cache()
    random.shuffle(val_datasets)
    import gc
    del Val_dataset
    del val_dataset
    gc.collect()
    return val_datasets

def Lalternate_training_datasets(model, Lmodel, Htraining_dataloader, opts):

    print('Generating lower-level training data')
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
        model.load_state_dict({**model.state_dict(), **{}.get('model', {})})
    if isinstance(Lmodel,torch.nn.DataParallel):
        Lmodel = Lmodel.module
        Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get('model', {})})
    model.eval()
    model.cuda()
    model.set_decode_type('greedy')
    training_datasets = []
    with torch.no_grad():
        for i in tqdm(Htraining_dataloader, disable=opts.no_progress_bar):
            if len(i) == 2:
                Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4, Lmask_dataset5, Lmask_dataset6 = model(move_to(i['data'], opts.device), Lmodel,
                                                                                       Lval_dataset=True)
            else:
                Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4, Lmask_dataset5, Lmask_dataset6 = model(move_to(i, opts.device), Lmodel, Lval_dataset=True)
            training_datasets.append(Lmask_dataset1)
            training_datasets.append(Lmask_dataset2)
            training_datasets.append(Lmask_dataset3)
            training_datasets.append(Lmask_dataset4)
            training_datasets.append(Lmask_dataset5)
            training_datasets.append(Lmask_dataset6)
    Htraining_dataloader = 0
    Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4, Lmask_dataset5, Lmask_dataset6 = 0, 0, 0, 0, 0, 0

    training_dataset = []
    for dataset in tqdm(training_datasets):
        for i in range(dataset['max_length'].shape[0]):
            data = {
                    'loc': dataset['loc'][i,:].cpu().detach(),
                    'prize': dataset['prize'][i,:].cpu().detach(),
                    'depot': dataset['depot'][i,:].cpu().detach(),
                    'max_length': dataset['max_length'][i].cpu().detach(),
                    'mask': dataset['mask'][i,:].copy()
                }
            training_dataset.append(data)

    torch.cuda.empty_cache()
    random.shuffle(training_dataset)
    import gc
    del training_datasets
    gc.collect()
    return training_dataset
