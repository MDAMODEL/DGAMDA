import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import sklearn
import pdb
from dgl.nn import GATv2Conv
from dgl.nn import GATConv
from dgl.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import argparse
from utils.plotauc import plot_auc_curves,plot_prc_curves
import os
import warnings
warnings.filterwarnings("ignore")
from torchsummary import summary

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_seed(seed=100):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# ----------- 2. create model -------------- #
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats,fc_out):
        super(GraphSAGE, self).__init__()
        self.mirnafc = nn.Linear(495, fc_out)
        self.diseasefc = nn.Linear(383, fc_out)
        self.conv1 = SAGEConv(fc_out, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        mirna_feats = in_feat[0:495, :]
        disease_feats = in_feat[495:878, 0:383]
        mirna_feats = self.mirnafc(mirna_feats)
        disease_feats = self.diseasefc(disease_feats)
        in_feat = torch.cat([mirna_feats, disease_feats], dim=0)
        # pdb.set_trace()

        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g,h)
        h = F.relu(h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, fc_out, num_heads):
        super(GAT, self).__init__()
        self.mirnafc = nn.Linear(495, fc_out)
        self.diseasefc = nn.Linear(383, fc_out)
        self.conv1 = GATConv(fc_out, h_feats, num_heads=num_heads)
        self.conv2 = GATConv(h_feats * num_heads, h_feats, num_heads=num_heads)

        self.relu = nn.ReLU()

    def forward(self, g, in_feat):
        mirna_feats = in_feat[0:495, :]
        disease_feats = in_feat[495:878, 0:383]
        mirna_feats = self.mirnafc(mirna_feats)
        disease_feats = self.diseasefc(disease_feats)
        in_feat = torch.cat([mirna_feats, disease_feats], dim=0)


        h = self.conv1(g, in_feat)
        h = self.relu(h)
        h = h.view(878, -1)
        h = self.conv2(g, h)
        h = self.relu(h)
        h = h.view(878, -1)
        return h
# build a two-layer GATAv2 model
class GATv2(nn.Module):
    def __init__(self, in_feats, h_feats, fc_out, num_heads):
        super(GATv2, self).__init__()
        self.mirnafc = nn.Linear(495, fc_out)
        self.diseasefc = nn.Linear(383, fc_out)
        self.conv1 = GATv2Conv(fc_out, h_feats, num_heads=num_heads)
        self.conv2 = GATv2Conv(h_feats * num_heads, h_feats, num_heads=num_heads)
        self.relu = nn.ReLU()

    def forward(self, g, in_feat):
        mirna_feats = in_feat[0:495, :]
        disease_feats = in_feat[495:878, 0:383]
        mirna_feats = self.mirnafc(mirna_feats)
        disease_feats = self.diseasefc(disease_feats)
        in_feat = torch.cat([mirna_feats, disease_feats], dim=0)


        h = self.conv1(g, in_feat)
        h = self.relu(h)
        h = h.view(878, -1)
        h = self.conv2(g, h)
        h = self.relu(h)
        h = h.view(878, -1)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * num_heads * 2, h_feats * num_heads)
        self.W2 = nn.Linear(h_feats * num_heads, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """

        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        score = self.W1(h)
        score = self.relu(score)
        score = self.W2(score)
        score = self.sigmoid(score).squeeze(1)
        return {'score': score}


    def forward(self, g, h):
        with g.local_scope():  #
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy(scores,labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)

    return roc_auc_score(labels, scores),fpr,tpr

def compute_acc_pre_recall_f1(pos_score,neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    label_val_cpu = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    pred_val = [0 if j < 0.5 else 1 for j in scores]
    acc_val = metrics.accuracy_score(label_val_cpu, pred_val)
    pre_val = metrics.precision_score(label_val_cpu, pred_val, average='binary')
    recall_val = metrics.recall_score(label_val_cpu, pred_val,average='binary')
    f1_val = metrics.f1_score(label_val_cpu, pred_val,average='binary')
    return acc_val,pre_val,recall_val,f1_val
def compute_prauc(pos_score,neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    label_val_cpu = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    precision, recall, _ = metrics.precision_recall_curve(label_val_cpu, scores)
    prauc = metrics.auc(recall, precision)
    return prauc,precision,recall

if __name__ == '__main__':
    set_seed()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    parser = argparse.ArgumentParser() 
    parser.add_argument("--epochs", default=100, type=int) 

    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--h_feats", default=32, type=int)
    parser.add_argument("--fc_out", default=128, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    args = parser.parse_args()
    print(args.num_heads)

    (g,), _  = dgl.load_graphs('data/graph_md_xsd_withoutgs_HGANMDA.dgl')



    u, v = g.edges()

    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids) 

    test_size = int(len(eids) * 0.2)
    train_size = g.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]


    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u), g.number_of_edges()) 

    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]


    test_pos_u_1, test_pos_v_1 = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u_1, train_pos_v_1 = u[eids[test_size:]], v[eids[test_size:]]

    test_pos_u_2, test_pos_v_2 = u[eids[test_size:test_size * 2]], v[eids[test_size:test_size * 2]]
    train_pos_u_2, train_pos_v_2 = u[np.hstack((eids[:test_size], eids[test_size * 2:]))], v[
        np.hstack((eids[:test_size], eids[test_size * 2:]))]

    test_pos_u_3, test_pos_v_3 = u[eids[test_size * 2:test_size * 3]], v[eids[test_size * 2:test_size * 3]]
    train_pos_u_3, train_pos_v_3 = u[np.hstack((eids[:test_size * 2], eids[test_size * 3:]))], v[
        np.hstack((eids[:test_size * 2], eids[test_size * 3:]))]

    test_pos_u_4, test_pos_v_4 = u[eids[test_size * 3:test_size * 4]], v[eids[test_size * 3:test_size * 4]]
    train_pos_u_4, train_pos_v_4 = u[np.hstack((eids[:test_size * 3], eids[test_size * 4:]))], v[
        np.hstack((eids[:test_size * 3], eids[test_size * 4:]))]

    test_pos_u_5, test_pos_v_5 = u[eids[test_size * 4:]], v[eids[test_size * 4:]]
    train_pos_u_5, train_pos_v_5 = u[eids[:test_size * 4]], v[eids[:test_size * 4]]


    test_neg_u_1, test_neg_v_1 = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u_1, train_neg_v_1 = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    test_neg_u_2, test_neg_v_2 = neg_u[neg_eids[test_size:test_size * 2]], neg_v[neg_eids[test_size:test_size * 2]]
    train_neg_u_2, train_neg_v_2 = neg_u[np.hstack((neg_eids[:test_size], neg_eids[test_size * 2:]))], neg_v[
        np.hstack((neg_eids[:test_size], neg_eids[test_size * 2:]))]

    test_neg_u_3, test_neg_v_3 = neg_u[neg_eids[test_size * 2:test_size * 3]], neg_v[
        neg_eids[test_size * 2:test_size * 3]]
    train_neg_u_3, train_neg_v_3 = neg_u[np.hstack((neg_eids[:test_size * 2], neg_eids[test_size * 3:]))], neg_v[
        np.hstack((neg_eids[:test_size * 2], neg_eids[test_size * 3:]))]

    test_neg_u_4, test_neg_v_4 = neg_u[neg_eids[test_size * 3:test_size * 4]], neg_v[
        neg_eids[test_size * 3:test_size * 4]]
    train_neg_u_4, train_neg_v_4 = neg_u[np.hstack((neg_eids[:test_size * 3], neg_eids[test_size * 4:]))], neg_v[
        np.hstack((neg_eids[:test_size * 3], neg_eids[test_size * 4:]))]

    test_neg_u_5, test_neg_v_5 = neg_u[neg_eids[test_size * 4:]], neg_v[neg_eids[test_size * 4:]]
    train_neg_u_5, train_neg_v_5 = neg_u[neg_eids[test_size * 4:]], neg_v[neg_eids[test_size * 4:]]

    train_pos_u_list = [train_pos_u_1,train_pos_u_2,train_pos_u_3,train_pos_u_4,train_pos_u_5]
    train_pos_v_list = [train_pos_v_1,train_pos_v_2,train_pos_v_3,train_pos_v_4,train_pos_v_5]
    train_neg_u_list = [train_neg_u_1,train_neg_u_2,train_neg_u_3,train_neg_u_4,train_neg_u_5]
    train_neg_v_list = [train_neg_v_1,train_neg_v_2,train_neg_v_3,train_neg_v_4,train_neg_v_5]
    test_pos_u_list = [test_pos_u_1,test_pos_u_2,test_pos_u_3,test_pos_u_4,test_pos_u_5]
    test_pos_v_list = [test_pos_v_1,test_pos_v_2,test_pos_v_3,test_pos_v_4,test_pos_v_5]
    test_neg_u_list = [test_neg_u_1,test_neg_u_2,test_neg_u_3,test_neg_u_4,test_neg_u_5]
    test_neg_v_list = [test_neg_v_1,test_neg_v_2,test_neg_v_3,test_neg_v_4,test_neg_v_5]
    eids_test_remove_list = [eids[:test_size],eids[test_size:2*test_size],eids[2*test_size:3*test_size],
                             eids[3*test_size:4*test_size],eids[4*test_size:]]
    prc_result = []  # pr-auc
    auc_result = [] # roc-auc
    acc_result = [] # acc
    pre_result = [] # precision
    recall_result = [] #recall
    f1_result = [] #f1

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    for kfold in range(5):
        train_g = dgl.remove_edges(g, eids_test_remove_list[kfold])

        train_g = dgl.add_self_loop(train_g)
        train_g.to(device)

        train_pos_g = dgl.graph((train_pos_u_list[kfold], train_pos_v_list[kfold]), num_nodes=g.number_of_nodes())
        train_neg_g = dgl.graph((train_neg_u_list[kfold], train_neg_v_list[kfold]), num_nodes=g.number_of_nodes())

        test_pos_g = dgl.graph((test_pos_u_list[kfold], test_pos_v_list[kfold]), num_nodes=g.number_of_nodes())
        test_neg_g = dgl.graph((test_neg_u_list[kfold], test_neg_v_list[kfold]), num_nodes=g.number_of_nodes())

        train_pos_g.to(device)
        train_neg_g.to(device)

        test_pos_g.to(device)
        test_neg_g.to(device)

        h_feats = args.h_feats
        fc_out = args.fc_out
        num_heads= args.num_heads
        model = GraphSAGE(train_g.ndata['feat'].shape[1], h_feats,fc_out) # AUC 0.958

        pred = MLPPredictor(h_feats) 
        pred.to(device)

        # ----------- 3. set up loss and optimizer -------------- #
        # in this case, loss will in training loop
        # weight_decay = 0.01
        # optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.005,
                                     weight_decay=1e-3)  # 0.965

        # ----------- 4. training -------------------------------- #
        all_logits = []
        auc = 0

        best_auc = 0
        best_acc = 0
        best_pr = 0
        best_acc = 0
        best_f1 = 0
        best_epoch = 0

        for e in range(100):
            # forward
            h = model(train_g, train_g.ndata['feat'])
            h = torch.squeeze(h)
            pos_score = pred(train_pos_g, h)
            neg_score = pred(train_neg_g, h)
            loss = compute_loss(pos_score, neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ----------- 5. check results ------------------------ #

            if e % 5 == 0:
                print('----------------')
 
                auc,_,_ = compute_auc(pos_score.detach(), neg_score.detach())
                acc, pre, recall, f1 = compute_acc_pre_recall_f1(pos_score.detach(), neg_score.detach())
                print('In epoch {}, loss: {:.4f}'.format(e, loss), 'Acc: %.4f' % acc, 'Pre: %.4f' % pre,
                      'Recall: %.4f' % recall, 'F1: %.4f' % f1,
                      'Train AUC: %.4f' %auc)

            model.eval()  
            with torch.no_grad():
                pos_score = pred(test_pos_g, h)
                neg_score = pred(test_neg_g, h)
                acc, pre, recall, f1 = compute_acc_pre_recall_f1(pos_score.detach(), neg_score.detach())
                auc,fpr,tpr = compute_auc(pos_score.detach(), neg_score.detach())
                if e % 5 == 0:
                    print('In epoch {}, loss: {:.4f}'.format(e, loss), 'Acc: %.4f' % acc, 'Pre: %.4f' % pre,
                          'Recall: %.4f' % recall, 'F1: %.4f' % f1,
                          'Test  AUC: %.4f' % auc)
                    print('----------------')
                auc,_,_ = compute_auc(pos_score.detach(), neg_score.detach())
                if  auc > best_auc:

                    best_auc,fpr,tpr = compute_auc(pos_score.detach(), neg_score.detach())
                    prc_auc,precision_val,recall_val = compute_prauc(pos_score.detach(), neg_score.detach())
                    best_acc = acc
                    best_pr = pre
                    best_recall = recall
                    best_f1 = f1
                    best_epoch = e

        print('In kfold {} epoch {},'.format(kfold,best_epoch),'Acc: %.4f' % best_acc, 'Pre: %.4f' % best_pr,
                              'Recall: %.4f' % best_recall, 'F1: %.4f' % best_f1,
                              'Test  AUC: %.4f' %best_auc)
        auc_result.append(best_auc)
        prc_result.append(prc_auc)
        fprs.append(fpr)
        tprs.append(tpr)

        acc_result.append(best_acc)
        pre_result.append(best_pr)
        recall_result.append(best_recall)
        f1_result.append(best_f1)

        precisions.append(precision_val)
        recalls.append(recall_val)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('epoch:{},'.format(args.epochs),'num_heads:{}'.format(args.num_heads),'fc_out:{}'.format(args.fc_out))
    print('ROC-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))
    print('-----------------------------------------------------------------------------------------------')
    plot_auc_curves(fprs, tprs, auc_result, directory='roc_result', name='test_auc')
    plot_prc_curves(precisions, recalls, prc_result, directory='roc_result', name='test_prc')