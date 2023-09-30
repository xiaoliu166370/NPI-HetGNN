import os

from utills.evaluation import *
from utills.utils import *
from model.my_model import Model

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.use_deterministic_algorithms(True)

data = open('dataset/NPR7317_processed/7317_r_p.txt', 'r')
links = data.readlines()
l = []
for link in links:
    link = link.strip()
    link = link.split(' ')
    link = [eval(ll) for ll in link]
    l.append(link)
links = np.array(l)
# train, test = split_train_test(links, 0.8)
# np.save('dataset/NPR7317_processed/train73178',train)
# np.save('dataset/NPR7317_processed/test73178',test)
train, test = np.load('dataset/NPR7317_processed/train73178.npy'), np.load('dataset/NPR7317_processed/test73178.npy')
# 生成训练图
link_src = train[:, 1]
link_dst = train[:, 0]
hetero_graph = dgl.heterograph({
    ('protein', 'link', 'rna'): (link_src, link_dst),
    ('rna', 'link_by', 'protein'): (link_dst, link_src),
})
print(hetero_graph)

# 生成测试图
t_link_src = test[:, 1]
t_link_dst = test[:, 0]
t_hetero_graph = dgl.heterograph({
    ('protein', 'link', 'rna'): (t_link_src, t_link_dst),
    ('rna', 'link_by', 'protein'): (t_link_dst, t_link_src),
})

test_links = np.stack([t_link_src, t_link_dst], axis=1)

# 节点特征
rna_features1 = torch.load('dataset/NPR7317_processed/rna7317_feature.pkl')
protein_features1 = torch.load('dataset/NPR7317_processed/proteinRPI7317_feature.pkl')
rna_features2 = torch.load('dataset/NPR7317_processed/RPIrna7317.emb')
protein_features2 = torch.load('dataset/NPR7317_processed/RPIprotein7317.emb')
rna_features = torch.cat((rna_features1, torch.tensor(rna_features2)), dim=1)
protein_features = torch.cat((protein_features1, torch.tensor(protein_features2)), dim=1)
# hetero_graph.nodes['rna'].data['feature'] = rna_features
hetero_graph.nodes['rna'].data['feature'] = torch.tensor(rna_features2)
# hetero_graph.nodes['protein'].data['feature'] = protein_features
hetero_graph.nodes['protein'].data['feature'] = torch.tensor(protein_features2)

k = 4
epoches = 100
lr = 2e-4
weight_decay = 0
device = 'cuda'

metapaths = [
    [["link", "link_by"], ["link", "link_by", "link", "link_by"]],
    [["link_by", "link"], ["link_by", "link", "link_by", "link"]]
]
model = Model([128, 128], 256, 256, 256, 4,2, 0.2, device, metapaths).to(device)
# model = Model([128, 128], 256, 256, 256, 4,2, 0.2, device, metapaths).to(device)
protein_feats = hetero_graph.nodes['protein'].data['feature']
rna_feats = hetero_graph.nodes['rna'].data['feature']
node_features = {'protein': protein_feats, 'rna': rna_feats}
opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

best = 0.0

# 需要保证测试集的负样本不在训练集的正样本和负样本中
t_negative_graph = construct_negative_graph(t_hetero_graph, 3, ('protein', 'link', 'rna'))
src_dst1 = t_negative_graph.edges(etype='link')
protein_id1 = src_dst1[0]
rna_id1 = src_dst1[1]
neg = torch.stack([protein_id1, rna_id1], dim=1).numpy()
neg1 = torch.stack([rna_id1, protein_id1], dim=1).numpy()
ll = []
for index, i in enumerate(neg1):
    dd = (i == links)
    d = np.sum(dd, axis=1)
    if 2 in d:
        ll.append(index)
neg = np.delete(neg, ll, axis=0)
negative_graph = construct_negative_graph(hetero_graph, k, ('protein', 'link', 'rna'))
num_neg_edges = negative_graph.num_edges(etype='link')
src_dst_neg = negative_graph.edges(etype='link')
protein_id_neg = src_dst_neg[0]
rna_id_neg = src_dst_neg[1]
neg11 = torch.stack([protein_id_neg, rna_id_neg], dim=1).numpy()
l_need_remove = []
for idx, n in enumerate(neg11):
    dd = (n == test_links)
    d = np.sum(dd, axis=1)
    if 2 in d:
        l_need_remove.append(idx)
n_r = num_neg_edges // k - len(l_need_remove)
negative_graph.remove_edges(l_need_remove)
num_neg_edges2 = negative_graph.num_edges(etype='link')
ridx = np.random.choice(num_neg_edges2, n_r, replace=False)
negative_graph.remove_edges(ridx)
src_dst_neg = negative_graph.edges(etype='link')
protein_id_neg = src_dst_neg[0]
rna_id_neg = src_dst_neg[1]
neg11 = torch.stack([protein_id_neg, rna_id_neg], dim=1).numpy()
lll = []
for index, i in enumerate(neg):
    dd = (i == neg11)
    d = np.sum(dd, axis=1)
    if 2 in d:
        lll.append(index)
neg_test = np.delete(neg, lll, axis=0)
test_samples = dict()
test_samples['pos_samples'] = np.concatenate((np.array(test[:, 1])[:, None], np.array(test[:, 0])[:, None]), axis=1)
test_samples['neg_samples'] = neg_test
# new_g1 = metapath_reachable_graph(hetero_graph, ["link", "link_by"])
# new_g2 = metapath_reachable_graph(hetero_graph, ["link_by", "link"])
# torch.save(new_g1,'dataset/NPR7317_processed/g7317_0_0.pkl')
# torch.save(new_g2,'dataset/NPR7317_processed/g7317_1_0.pkl')
# g_list = [[torch.load('dataset/NPR7317_processed/g7317_0_0.pkl'),torch.load('dataset/NPR7317_processed/g7317_0_1.pkl')],[torch.load('dataset/NPR7317_processed/g7317_1_0.pkl'),torch.load('dataset/NPR7317_processed/g7317_1_1.pkl')]]
# edge_l = [torch.load('dataset/NPR7317_processed/edge7317_0_1.pkl'),torch.load('dataset/NPR7317_processed/edge7317_1_1.pkl')]
# torch.save(g_list,'g_list7317.pkl')
# torch.save(edge_l,'edge_l7317.pkl')
g_list = torch.load('dataset/NPR7317_processed/g_list7317.pkl')
edge_l = torch.load('dataset/NPR7317_processed/edge_l7317.pkl')
for epoch in range(epoches):

    pos_score, neg_score, h = model(g_list, hetero_graph, negative_graph, node_features, ('protein', 'link', 'rna'),
                                    edge_l)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()

    embs = dict()
    embs['protein'] = h['protein'].cpu().detach()
    embs['rna'] = h['rna'].cpu().detach()
    avg_auroc, std_auroc, avg_auprc, std_auprc, acc, avg_recall, avg_spe, avg_pre, avg_mcc = link_prediction(epoch+1,embs,
                                                                                                             test_samples,
                                                                                                             'hadamard')

    if avg_mcc > best:
        best = avg_mcc
        torch.save(h['protein'], 'protein.emb')
        torch.save(h['rna'], 'rna.emb')
        print(
            "### epoch: {}, loss: {}, Average(over trials) of NPI-HetGNN:  AUROC: {:.4f}({:.4f}), AUPRC: {:.4f}({:.4f}), ACC: {:.4f},Recall: {:.4f}, Specifity: {:.4f},Precision: {:.4f},MCC: {:.4f}".format(
                epoch + 1,
                loss.item(),
                avg_auroc,
                std_auroc,
                avg_auprc,
                std_auprc,
                acc,
                avg_recall,
                avg_spe,
                avg_pre,
                avg_mcc))
