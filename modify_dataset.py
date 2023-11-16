import numpy as np
import torch
import math
import random

# --------------------------------- RANDOM PAIRS GENERATION ---------------------------------
def decode(i):
    k = math.floor((1+math.sqrt(1+8*i))/2)  # eq1
    return k,i-k*(k-1)//2

def rand_pair(n):
    return decode(random.randrange(n*(n-1)//2))

def rand_pairs(n,m):
    return [decode(i) for i in random.sample(range(n*(n-1)//2),m)]

# ----------------------------------- MATRIX PERTURBATION -----------------------------------
def pert_matrix(arr):
    """Perturbs 1% of the values in the matrix (excludes diagonal elements)

    Args:
        arr (numpy ndarray): matrix to be perturbed

    Returns:
        numpy ndarray: perturbed matrix
    """
    num_val_perturb = int(0.01*arr.shape[0]*arr.shape[1])

    rand_nos = rand_pairs(min(arr.shape[0],arr.shape[1]), num_val_perturb)

    for i in range(num_val_perturb):
        p = rand_nos[i][0]
        q = rand_nos[i][1]
        # Note that p==q never happends due to the decode method
        # If k = i - k(k-1)//2 then k won't satisfy the eq1
        arr[p][q] = 1.0 - arr[p][q]
    
    return arr

#  ------------------------- DATASET MODIFICATION BEFORE EXPLANATION ------------------------
def modify(dict, mode="identical"):
    modified_adj = np.array(dict['adj'])
    modified_adj = np.reshape(modified_adj,dict['adj'].shape)

    modified_feat = np.array(dict['feat'])
    modified_feat = np.reshape(modified_feat,dict['feat'].shape)

    modified_label = np.array(dict['label'])
    modified_label = np.reshape(modified_label,dict['label'].shape)

    modified_pred = np.array(dict['pred'])
    modified_pred = np.reshape(modified_pred,dict['pred'].shape)

    modified_train_idx = np.array([dict['train_idx']])

    # Duplicate the values in all the arrays
    modified_adj = np.repeat(modified_adj,2)
    modified_adj = np.reshape(modified_adj,(dict['adj'].shape[0]*2, dict['adj'].shape[1], dict['adj'].shape[2]))

    modified_feat = np.repeat(modified_feat,2)
    modified_feat = np.reshape(modified_feat,(dict['feat'].shape[0]*2, dict['feat'].shape[1], dict['feat'].shape[2]))

    modified_label = np.repeat(modified_label,2)

    modified_pred = np.repeat(modified_pred,2)
    modified_pred = np.reshape(modified_pred,(dict['pred'].shape[0], dict['pred'].shape[1]*2, dict['pred'].shape[2]))

    modified_train_idx = np.repeat(modified_train_idx,2)

    if mode=="similar":
        for i in range(1,len(modified_adj),2):
            modified_adj[i] = pert_matrix(modified_adj[i])

    dict1 = {}
    dict1['adj'] = torch.Tensor(modified_adj)
    dict1['feat'] = torch.Tensor(modified_feat)
    dict1['label'] = torch.Tensor(modified_label)
    dict1['pred'] = torch.Tensor(modified_pred)
    dict1['train_idx'] = list(modified_train_idx)

    # # Cross-checking shape
    # for p in list(dict1.keys()):
    #     if p != 'train_idx':
    #         print(f"{p}: {dict1[p].shape}")

    return dict1

def main():
    adj = torch.Tensor([np.ones((25,25))]*100)
    feat = torch.Tensor([np.ones((25,10))]*100)
    label = torch.Tensor([1]*10)
    pred = torch.Tensor(np.ones((1,100,2)))
    train_idx = list(range(6))

    dict = {'adj':adj, 'feat':feat, 'label':label, 'pred':pred, 'train_idx':train_idx}

    dict1 = modify(dict)
    dict2 = modify(dict,"similar")

    print(f"Identical: {dict1}")
    print(f"Similar: {dict2['adj']}")

if __name__ == "__main__":
    main()