import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
from rec.model.pinsage import PinSage
#from rec.datasets.ccf_ai import ccf_ai
from rec.utils import cuda
from dgl import DGLGraph

import argparse
import pickle
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--learning_option', type=str, default='SGD')
parser.add_argument('--lr', type=float, default=1)
parser.add_argument('--pre_set', type=str, default='none')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--use-feature', action='store_true')
parser.add_argument('--sgd-switch', type=int, default=-1)
parser.add_argument('--n-negs', type=int, default=10)
parser.add_argument('--loss', type=str, default='hinge')
parser.add_argument('--hard-neg-prob', type=float, default=0.1)
parser.add_argument('--decay_factor', type=float, default=0.98)
parser.add_argument('--suffix', type=str, default='')
parser.add_argument('--zero_h', action='store_true')
args = parser.parse_args()

print(args)

inputnetwork = 'collect.pkl'

if os.path.exists(inputnetwork):
    with open(inputnetwork, 'rb') as f:
        dataset = pickle.load(f)
else:
   print("no data found")
###
####
graph = dataset.g
neighbors = dataset.neighbors.to(dtype=torch.long)

hidden_number = 256
layer_number = args.layers
batch_size = 256
margin = 0.9

negative_samples = args.n_negs
hard_negative_probabilities = np.linspace(0, args.hard_neg_prob, args.epochs)

parameters = {
        'none': lambda epoch: 1,
        'decay': lambda epoch: max(args.decay_factor ** epoch, 1e-4),
        }
lossfunction = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

inputfeatures = hidden_number
embeddings = nn.ModuleDict()
embeddings['year'] = nn.Embedding(
    graph.ndata['year'].max().item() + 1,
    inputfeatures,
    padding_idx=0
        )
if 'venue' in graph.ndata.keys():
    embeddings['venue'] = nn.Embedding(
        graph.ndata['venue'].max().item() + 1,
        inputfeatures,
        padding_idx=0
            )
embeddings['fos'] = nn.Sequential(
    nn.Linear(300, inputfeatures),
    nn.LeakyReLU(),
    )

model = cuda(PinSage(
    graph.number_of_nodes(),
    [hidden_number] * (layer_number + 1),
    20,
    0.5,
    20,
    emb=embeddings,
    G=graph,
    zero_h=args.zero_h
    ))
learning_option = getattr(torch.optim, args.learning_option)(model.parameters(), lr=args.lr)
pre_set = torch.optim.lr_scheduler.LambdaLR(learning_option, parameters[args.pre_set])


def forward(model, pre_graph, nodeset, train=True):
    if train:
        return model(pre_graph, nodeset)
    else:
        with torch.no_grad():
            return model(pre_graph, nodeset)


def filter_nid(node_ids, nid_from):
    node_ids = [nid.numpy() for nid in node_ids]
    nid_from = nid_from.numpy()
    np_mask = np.logical_and(*[np.isin(node_id, nid_from) for node_id in node_ids])
    return [torch.from_numpy(node_id[np_mask]) for node_id in node_ids]


def train_batches(pre_graph_edges, train_graph_edges, train):
    global learning_option
    if train:
        model.train()
    else:
        model.eval()

    pre_graph_source, pre_graph_destination = graph.find_edges(pre_graph_edges)
    pre_graph = DGLGraph()
    pre_graph.add_nodes(graph.number_of_nodes())
    pre_graph.add_edges(pre_graph_source, pre_graph_destination)
    pre_graph.ndata.update({k: cuda(v) for k, v in graph.ndata.items()})
    edge_batches = train_graph_edges[torch.randperm(train_graph_edges.shape[0])].split(batch_size)

    with tqdm.tqdm(edge_batches) as tq:
        loss_num = 0
        acc_num = 0
        i = 0
        for batch_id, batch in enumerate(tq):
            i += batch.shape[0]
            source, detination = graph.find_edges(batch)
            destination_negatives = []
            for i in range(len(detination)):
                if np.random.rand() < args.hard_neg_prob:
                    neighbor = torch.LongTensor(neighbors[detination[i].item()])
                    mask = ~(graph.has_edges_between(neighbor, source[i].item()).byte())
                    destination_negatives.append(np.random.choice(neighbor[mask].numpy(), negative_samples))
                else:
                    destination_negatives.append(np.random.randint(
                        0, len(dataset.papers), negative_samples))


            destination_negatives = torch.LongTensor(destination_negatives)
            detination = detination.view(-1, 1).expand_as(destination_negatives).flatten()
            source = source.view(-1, 1).expand_as(destination_negatives).flatten()
            destination_negatives = destination_negatives.flatten()

            mask = (pre_graph.in_degrees(destination_negatives) > 0) & \
                   (pre_graph.in_degrees(detination) > 0) & \
                   (pre_graph.in_degrees(source) > 0)
            source = source[mask]
            detination = detination[mask]
            destination_negatives = destination_negatives[mask]
            if len(source) == 0:
                continue

            nodeset = cuda(torch.cat([source, detination, destination_negatives]))
            source_size, destination_size, negative_destination_size = \
                    source.shape[0], detination.shape[0], destination_negatives.shape[0]

            hidden_source, hidden_destination, negative_hidden_destination = (
                    forward(model, pre_graph, nodeset, train)
                    .split([source_size, destination_size, negative_destination_size]))

            difference = (hidden_source * (negative_hidden_destination - hidden_destination)).sum(1)
            loss = lossfunction[args.loss](difference)
            accuracy = (difference < 0).sum()
            assert loss.item() == loss.item()

            grad_sqr_norm = 0
            if train:
                learning_option.zero_grad()
                loss.backward()
                for name, p in model.named_parameters():
                    assert (p.grad != p.grad).sum() == 0
                    grad_sqr_norm += p.grad.norm().item() ** 2
                learning_option.step()

            loss_num += loss.item()
            acc_num += accuracy.item() / negative_samples
            avg_loss = loss_num / (batch_id + 1)
            average_accuracy = acc_num / i
            tq.set_postfix({'loss': '%.6f' % loss.item(),
                            'avg_loss': '%.3f' % avg_loss,
                            'average_accuracy': '%.3f' % average_accuracy,
                            'grad_norm': '%.6f' % np.sqrt(grad_sqr_norm)})

    return avg_loss, average_accuracy

def nodeidremap(paperids, offset, p):
    hiddenrepresentation_id = paperids[np.where((paperids >= offset) & ((paperids - offset) % p == 0))]
    return (hiddenrepresentation_id - offset) // p

def testing(pre_graph_edges, epoch, validation=True):
    model.eval()
    period = 1
    offset = epoch % period
    number_of_users = len(dataset.authors.index)
    number_of_items = len(dataset.papers.index)

    pre_graph_source, pre_graph_destination = graph.find_edges(pre_graph_edges)
    pre_graph = DGLGraph()
    pre_graph.add_nodes(graph.number_of_nodes())
    pre_graph.add_edges(pre_graph_source, pre_graph_destination)
    pre_graph.ndata.update({k: cuda(v) for k, v in graph.ndata.items()})

    user_offset = 0
    hiddenrepresentationlist = []
    with torch.no_grad():
        with tqdm.trange(offset, number_of_users + number_of_items, period) as tq:
            for node_id in tq:
                if user_offset == 0 and node_id >= number_of_items:
                    user_offset = node_id

                nodeset = cuda(torch.LongTensor([node_id]))
                hiddenrepresentation = forward(model, pre_graph, nodeset, False)
                hiddenrepresentationlist.append(hiddenrepresentation)
    hiddenrepresentation = torch.cat(hiddenrepresentationlist, 0)

    rankinglist = []

    with torch.no_grad():
        with tqdm.trange(user_offset, number_of_items + number_of_users, period) as tq:
            for u_nid in tq:
                # userid = dataset.user_ids[u_nid]
                userid = u_nid
                uhid = (u_nid - offset)//period

                paperids_excluded = dataset.links[
                    (dataset.links['idx_A'] == userid) &
                    (dataset.links['train'] | dataset.links['test' if validation else 'valid'])
                    ]['idx_P'].values
                papaer_ids_candidate = dataset.links[
                    (dataset.links['idx_A'] == userid) &
                    dataset.links['valid' if validation else 'test']
                    ]['idx_P'].values

                paper_ids = np.setdiff1d(range(len(dataset.paper_ids_map)), paperids_excluded)

                hidden_representation_ids = nodeidremap(paper_ids, offset, period)
                hidden_representation_ids= hidden_representation_ids/1.0
                hidddens_candidate = nodeidremap(papaer_ids_candidate, offset, period)

                destination = torch.from_numpy(hidden_representation_ids/1.0).type(torch.long)
                source = torch.zeros_like(destination).fill_(uhid)
                hidden_destination = hiddenrepresentation[destination]
                hidden_source = hiddenrepresentation[source]

                score = (hidden_source * hidden_destination).sum(1)
                score_sort_idx = score.sort(descending=True)[1].cpu().numpy()

                rank_map = {v: i for i, v in enumerate(hidden_representation_ids[score_sort_idx])}
                rank_candidates = np.array([rank_map[p_nid] for p_nid in hidddens_candidate])
                rank = 1 / (rank_candidates + 1) if len(rank_candidates)!= 0 else np.array([1/ len(score_sort_idx)])
                rankinglist.append(rank.mean())
                tq.set_postfix({'rank': rank.mean()})

    return np.array(rankinglist)


def train():
    global learning_option, pre_set
    log_val = open(f'val_{args.suffix}.log', 'w')
    log_test = open(f'test_{args.suffix}.log', 'w')
    log_train = open(f'train_{args.suffix}.log', 'w')
    best_mrr = 0
    for epoch in range(args.epochs):
        args.hard_neg_prob = hard_negative_probabilities[epoch]
        dataset.refresh_mask()
        g_prior_edges = graph.filter_edges(lambda edges: edges.data['prior'])
        g_train_edges = graph.filter_edges(lambda edges: edges.data['train'] & ~edges.data['inv'])
        g_prior_train_edges = graph.filter_edges(
                lambda edges: edges.data['prior'] | edges.data['train'])
        #
        if (epoch+1)%10==0:
            print('Epoch %d validation' % epoch)
            with torch.no_grad():
                valid_mrr = testing(g_prior_train_edges, epoch, True)
                log_val.write(f'{valid_mrr.mean()}\n')
                if best_mrr < valid_mrr.mean():
                    best_mrr = valid_mrr.mean()
                    torch.save(model.state_dict(), f'model_best_{args.suffix}.pt')
            print(pd.Series(valid_mrr).describe())
            print('Epoch %d test' % epoch)
            with torch.no_grad():
                test_mrr = testing(g_prior_train_edges, False)
                log_test.write(f"{test_mrr.mean()}\n")
            print(pd.Series(test_mrr).describe())

        print('Epoch %d train' % epoch)
        avg_loss, avg_acc = train_batches(g_prior_edges, g_train_edges, True)
        log_train.write(f'{avg_loss} {avg_acc}\n')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_itr{epoch}_{args.suffix}.pt')

        if epoch == args.sgd_switch:
            learning_option = torch.optim.SGD(model.parameters(), lr=0.05)
            pre_set = torch.optim.lr_scheduler.LambdaLR(learning_option, parameters['decay'])
        # elif epoch < args.sgd_switch:
        pre_set.step()
    log_train.close()
    log_val.close()
    log_test.close()


if __name__ == '__main__':
    train()
