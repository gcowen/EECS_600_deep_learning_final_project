import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
from rec.datasets.ccf_ai import ccf_ai
from rec.utils import cuda
from dgl import DGLGraph
import matplotlib.pyplot as plt
import argparse
import pickle
import os

inputnetwork = 'collect.pkl'
with open(inputnetwork, 'rb') as f:
    datasets = pickle.load(f)

graph = datasets.g
hiddenrepresentation = torch.load('trained_results.pt')
offset = 0
period = 1
user_offset = len(datasets.papers)

def id_remap(paperids, offset, p):
    hiddenrepresentation = paperids[np.where((paperids >= offset) & ((paperids - offset) % p == 0))]
    return (hiddenrepresentation - offset) // p


def recommendation(user):
    userid = user_offset + user
    u_nid = userid
    user_id_in_represenration = (u_nid - offset)//period

    pids_exclude = datasets.links[
        (datasets.links['idx_A'] == userid) &
        (datasets.links['train'] | datasets.links['test'])
        ]['idx_P'].values
    pids_candidate = datasets.links[
        (datasets.links['idx_A'] == userid) &
        datasets.links['valid']
        ]['idx_P'].values

    papaerids = np.setdiff1d(range(len(datasets.paper_ids_map)), pids_exclude)

    hiddenrepresentations = id_remap(papaerids, offset, period)
    hids_candidate = id_remap(pids_candidate, offset, period)

    hiddenrepresentations=hiddenrepresentations/1.0
    destinations = torch.from_numpy(hiddenrepresentations).type(torch.long)
    sources = torch.zeros_like(destinations).fill_(user_id_in_represenration)
    print(destinations.dtype)
    hiddendestinations = hiddenrepresentation[destinations]
    hiddencources = hiddenrepresentation[sources]

    score = (hiddencources * hiddendestinations).sum(1)
    score_sort_idx = score.sort(descending=True)[1].cpu().numpy()
    return score_sort_idx,hiddenrepresentations/1.0

def main():
    while True:
        author_userid = input("Input: the index of author :")
        author_userid = eval(author_userid)
        print("The publications of author ", author_userid,":")
        print(datasets.papers.iloc[datasets.author_write[author_userid]][['title', 'venue', 'year']])
        score_sort_idx,hids= recommendation(author_userid)
        print("Recommendation:")
        print(datasets.papers.iloc[hids[score_sort_idx[:20]]][['title', 'venue', 'year']])

if __name__ == '__main__':
    main()

