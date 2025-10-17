import os

from torch.utils.data import DataLoader
from utils import load_model
from op_calculate.utils import load_model as Lload_model
from matplotlib import pyplot as plt
import torch
import numpy as np
import tqdm
import time


def allocation_scheme_plot(init_dataset, tour_1, tour_2, tour_3, tour_4):
    plt.figure(figsize=(6, 6))
    allocation_scheme = np.concatenate((tour_1, tour_2, tour_3, tour_4), axis=0)
    for single_scheme in allocation_scheme:
        plt.scatter(
            init_dataset[single_scheme, 0], init_dataset[single_scheme, 1], s=20
        )
        plt.scatter(init_dataset[0, 0], init_dataset[0, 1], c="r", marker="s", s=80)
    plt.show(block=True)


def allo_schedule_scheme_plot(init_dataset, allo_scheme, schedu_scheme):
    plt.figure(figsize=(6, 6))
    allocation_scheme = np.concatenate(allo_scheme, axis=0)
    for i in range(allocation_scheme.shape[0]):
        plt.scatter(
            init_dataset[allocation_scheme[i], 0],
            init_dataset[allocation_scheme[i], 1],
            s=20,
        )
        plt.scatter(init_dataset[0, 0], init_dataset[0, 1], c="r", marker="s", s=80)
        plt.plot(schedu_scheme[i][:, 0], schedu_scheme[i][:, 1])
    plt.show(block=True)


model_dir = {
    80: [
        "./outputs/taop_80/run_20211031T071011/epoch-99.pt",
        "./outputs/op_80/run_20211031T071011/epoch-1099.pt",
    ],
    100: [
        "./outputs/taop_100/run_20211104T015058/epoch-98.pt",
        "./outputs/op_100/run_20211104T015058/epoch-1098.pt",
    ],
    150: [
        "./outputs/taop_150/run_20211115T061900/epoch-96.pt",
        "./outputs/op_150/run_20211115T061900/epoch-1099.pt",
    ],
    200: [
        "./outputs/taop_200/run_20211124T124900/epoch-97.pt",
        "./outputs/op_200/run_20211124T124900/epoch-1097.pt",
    ],
    300: [
        "./outputs/taop_300/run_20211222T092337/epoch-101.pt",
        "./outputs/op_300/run_20211222T092337/epoch-1101.pt",
    ],
    500: [
        "./outputs/taop_500/run_20220203T050524/epoch-95.pt",
        "./outputs/op_500/run_20220203T050524/epoch-1095.pt",
    ],
}

problem_size = [80]
return_allocate = True
return_pi = True

for size in problem_size:
    num_samples = 1

    model, _ = load_model(model_dir[size][0])
    Lmodel, _ = Lload_model(model_dir[size][1])
    torch.manual_seed(1111)
    model.cuda()
    dataset = [
        {
            "loc": torch.FloatTensor(size, 2).uniform_(0, 1).cuda(),
            "depot": torch.FloatTensor(2).uniform_(0, 1).cuda(),
            "prize": torch.ones(size).cuda(),
            "max_length": torch.tensor(2.0).cuda(),
        }
        for i in range(num_samples)
    ]
    Dataset = DataLoader(dataset, batch_size=500)
    batch = next(iter(Dataset))
    model.eval()
    Lmodel.eval()
    model.set_decode_type("greedy")
    start_time = time.time()

    if return_pi and return_allocate:
        with torch.no_grad():
            allocate_scheme, schedule_scheme = model(
                batch, Lmodel, return_allocate=return_allocate, return_pi=return_pi
            )

        dataset_ = np.concatenate(
            (
                dataset[0]["depot"].cpu().numpy()[None, :],
                dataset[0]["loc"].cpu().numpy(),
            ),
            axis=0,
        )
        allo_scheme_ = [x.cpu().numpy() for x in allocate_scheme]
        schedule_scheme_ = [
            np.concatenate(
                (dataset[0]["depot"].cpu().numpy()[None, ...], np.squeeze(x, 0))
            )
            for x in schedule_scheme
        ]
        allo_schedule_scheme_plot(dataset_, allo_scheme_, schedule_scheme_)
    elif return_allocate:
        with torch.no_grad():
            tour_1, tour_2, tour_3, tour_4 = model(
                batch, Lmodel, return_allocate=return_allocate
            )

        dataset_ = np.concatenate(
            (
                dataset[0]["depot"].cpu().numpy()[None, :],
                dataset[0]["loc"].cpu().numpy(),
            ),
            axis=0,
        )
        # plt.scatter(dataset_[:, 0], dataset_[:, 1])
        # plt.show()
        allocation_scheme_plot(
            dataset_,
            tour_1.cpu().numpy(),
            tour_2.cpu().numpy(),
            tour_3.cpu().numpy(),
            tour_4.cpu().numpy(),
        )
