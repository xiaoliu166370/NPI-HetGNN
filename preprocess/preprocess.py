import pathlib
import pickle
import numpy as np

np.random.seed(1)

save_prefix = '../dataset/NPR7317_processed/'

l = np.load('train73178.npy')
num_proteins = 118
num_rna = 1874

dim = num_proteins + num_rna
adjM = np.zeros((dim, dim), dtype=int)

for r, p in l:
    pid = p
    rid = num_proteins + r
    adjM[pid, rid] = 1
    adjM[rid, pid] = 1

protein_rna_list = {i: adjM[i, num_proteins:num_proteins + num_rna].nonzero()[0] for i in range(num_proteins)}
rna_protein_list = {i: adjM[num_proteins + i, :num_proteins].nonzero()[0] for i in range(num_rna)}

# 0-1-0
p_r_p = []
for r, p_list in rna_protein_list.items():
    p_r_p.extend([(p1, r, p2) for p1 in p_list for p2 in p_list])
p_r_p = np.array(p_r_p)
p_r_p[:, 1] += num_proteins
sorted_index = sorted(list(range(len(p_r_p))), key=lambda i: p_r_p[i, [0, 2, 1]].tolist())  # 重新排列，优先级为0，2，1
p_r_p = p_r_p[sorted_index]
print('prp.shape', p_r_p.shape)

# 1-0-1
r_p_r = []
for p, r_list in protein_rna_list.items():
    r_p_r.extend([(r1, p, r2) for r1 in r_list for r2 in r_list])
r_p_r = np.array(r_p_r)
r_p_r[:, [0, 2]] += num_proteins
sorted_index = sorted(list(range(len(r_p_r))), key=lambda i: r_p_r[i, [0, 2, 1]].tolist())  # 重新排列，优先级为0，2，1
r_p_r = r_p_r[sorted_index]
print('rpr.shape', r_p_r.shape)
# 0-1-0-1-0
# r_p_r_idx = np.random.choice(len(r_p_r), int(0.2 * len(r_p_r)), replace=False)
# r_p_r_ = r_p_r[r_p_r_idx]
p_r_p_r_p = []
for r1, p, r2 in r_p_r:
    if len(rna_protein_list[r1 - num_proteins]) == 0 or len(rna_protein_list[r2 - num_proteins]) == 0:
        continue
    # candidate_p0_list = np.random.choice(len(rna_protein_list[r1 - num_proteins]),
    #                                      int(0.2 * len(rna_protein_list[r1 - num_proteins])), replace=False)

    candidate_p0_list = rna_protein_list[r1 - num_proteins]  # [candidate_p0_list]

    # candidate_p2_list = np.random.choice(len(rna_protein_list[r2 - num_proteins]),
    #                                      int(0.2 * len(rna_protein_list[r2 - num_proteins])), replace=False)
    candidate_p2_list = rna_protein_list[r2 - num_proteins]  # [candidate_p2_list]

    p_r_p_r_p.extend([(p0, r1, p, r2, p2) for p0 in candidate_p0_list for p2 in candidate_p2_list])

p_r_p_r_p = np.array(p_r_p_r_p)
sorted_index = sorted(list(range(len(p_r_p_r_p))), key=lambda i: p_r_p_r_p[i, [0, 4, 1, 2, 3]].tolist())
p_r_p_r_p = p_r_p_r_p[sorted_index]
print('prprp.shape', p_r_p_r_p.shape)
# 1-0-1-0-1
r_p_r_p_r = []
# p_r_p_idx = np.random.choice(len(p_r_p), int(0.2 * len(p_r_p)), replace=False)
# p_r_p_ = p_r_p[p_r_p_idx]
for p1, r, p2 in p_r_p:
    if len(protein_rna_list[p1]) == 0 or len(protein_rna_list[p2]) == 0:
        continue
    candidate_r0_list = np.random.choice(len(protein_rna_list[p1]),
                                         int(0.2 * len(protein_rna_list[p1])), replace=False)

    candidate_r0_list = protein_rna_list[p1][candidate_r0_list]

    candidate_r2_list = np.random.choice(len(protein_rna_list[p2]),
                                         int(0.2 * len(protein_rna_list[p2])), replace=False)
    candidate_r2_list = protein_rna_list[p2][candidate_r2_list]

    r_p_r_p_r.extend([(r0, p1, r, p2, r2) for r0 in candidate_r0_list for r2 in candidate_r2_list])

r_p_r_p_r = np.array(r_p_r_p_r)
r_p_r_p_r[:, [0, 4]] += num_proteins
sorted_index = sorted(list(range(len(r_p_r_p_r))), key=lambda i: r_p_r_p_r[i, [0, 4, 1, 2, 3]].tolist())
r_p_r_p_r = r_p_r_p_r[sorted_index]
print('rprpr.shape', r_p_r_p_r.shape)
expected_metapaths = [
    [(0, 1, 0), (0, 1, 0, 1, 0)],
    [(1, 0, 1), (1, 0, 1, 0, 1)]
]

for i in range(len(expected_metapaths)):
    pathlib.Path(save_prefix + '{}'.format(i)).mkdir(parents=True, exist_ok=True)

metapath_indices_mapping = {(0, 1, 0): p_r_p,
                            (0, 1, 0, 1, 0): p_r_p_r_p,
                            (1, 0, 1): r_p_r,
                            (1, 0, 1, 0, 1): r_p_r_p_r}

target_idx_lists = [np.arange(num_proteins), np.arange(num_rna)]
offset_list = [0, num_proteins]
for i, metapaths in enumerate(expected_metapaths):
    for metapath in metapaths:
        edge_metapath_idx_array = metapath_indices_mapping[metapath]

        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '_idx.pickle', 'wb') as out_file:
            target_metapaths_mapping = {}
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
                left = right
            pickle.dump(target_metapaths_mapping, out_file)

        with open(save_prefix + '{}/'.format(i) + '-'.join(map(str, metapath)) + '.adjlist', 'w') as out_file:
            left = 0
            right = 0
            for target_idx in target_idx_lists[i]:
                while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + \
                        offset_list[i]:
                    right += 1
                neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
                neighbors = list(map(str, neighbors))
                if len(neighbors) > 0:
                    out_file.write('{} '.format(target_idx) + ' '.join(neighbors) + '\n')
                else:
                    out_file.write('{}\n'.format(target_idx))
                left = right
