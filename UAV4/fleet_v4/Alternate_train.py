import gc
import torch
from torch.utils.data import DataLoader
from utils import move_to
from tqdm import tqdm
import random


def Lalternate_val_dataset(model, Lmodel, val_dataset, opts):
    print("Generating the validation data for the lower level model")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        model.load_state_dict({**model.state_dict(), **{}.get("model", {})})
    if isinstance(Lmodel, torch.nn.DataParallel):
        Lmodel = Lmodel.module
        Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get("model", {})})
    model.eval()
    model.cuda()
    model.set_decode_type("greedy")
    Val_dataset = DataLoader(
        val_dataset, batch_size=opts.eval_batch_size, num_workers=0
    )
    val_datasets = []
    with torch.no_grad():
        for i in tqdm(Val_dataset, disable=opts.no_progress_bar):
            Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4 = model(
                move_to(i, opts.device), Lmodel, Lval_dataset=True
            )
            for i in range(Lmask_dataset1["max_length"].shape[0]):
                data1 = {
                    "loc": Lmask_dataset1["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset1["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset1["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset1["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset1["mask"][i, :].copy(),
                }
                val_datasets.append(data1)

                data2 = {
                    "loc": Lmask_dataset2["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset2["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset2["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset2["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset2["mask"][i, :].copy(),
                }
                val_datasets.append(data2)

                data3 = {
                    "loc": Lmask_dataset3["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset3["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset3["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset3["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset3["mask"][i, :].copy(),
                }
                val_datasets.append(data3)

                data4 = {
                    "loc": Lmask_dataset4["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset4["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset4["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset4["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset4["mask"][i, :].copy(),
                }
                val_datasets.append(data4)

    torch.cuda.empty_cache()
    random.shuffle(val_datasets)
    del Val_dataset
    gc.collect()
    return val_datasets


def Lalternate_training_datasets(model, Lmodel, Htraining_dataloader, opts):

    print("Generating the training data for the lower level model")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        model.load_state_dict({**model.state_dict(), **{}.get("model", {})})
    if isinstance(Lmodel, torch.nn.DataParallel):
        Lmodel = Lmodel.module
        Lmodel.load_state_dict({**Lmodel.state_dict(), **{}.get("model", {})})
    model.eval()
    Lmodel.eval()
    model.cuda()
    Lmodel.cuda()
    model.set_decode_type("greedy")
    # training_datasets = []
    training_dataset = []
    with torch.no_grad():
        for i in tqdm(Htraining_dataloader, disable=opts.no_progress_bar):
            if len(i) == 2:
                Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4 = model(
                    move_to(i["data"], opts.device), Lmodel, Lval_dataset=True
                )
            else:
                Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4 = model(
                    move_to(i, opts.device), Lmodel, Lval_dataset=True
                )
            for i in range(Lmask_dataset1["max_length"].shape[0]):
                data1 = {
                    "loc": Lmask_dataset1["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset1["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset1["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset1["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset1["mask"][i, :].copy(),
                }
                training_dataset.append(data1)

                data2 = {
                    "loc": Lmask_dataset2["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset2["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset2["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset2["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset2["mask"][i, :].copy(),
                }
                training_dataset.append(data2)

                data3 = {
                    "loc": Lmask_dataset3["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset3["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset3["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset3["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset3["mask"][i, :].copy(),
                }
                training_dataset.append(data3)

                data4 = {
                    "loc": Lmask_dataset4["loc"][i, :].cpu().detach(),
                    "prize": Lmask_dataset4["prize"][i, :].cpu().detach(),
                    "depot": Lmask_dataset4["depot"][i, :].cpu().detach(),
                    "max_length": Lmask_dataset4["max_length"][i].cpu().detach(),
                    "mask": Lmask_dataset4["mask"][i, :].copy(),
                }
                training_dataset.append(data4)
    Htraining_dataloader = 0
    Lmask_dataset1, Lmask_dataset2, Lmask_dataset3, Lmask_dataset4 = 0, 0, 0, 0
    torch.cuda.empty_cache()
    random.shuffle(training_dataset)
    return training_dataset
