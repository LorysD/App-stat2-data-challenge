import numpy as np
import re
from scipy.sparse import coo_matrix
import time

# one-hot-encoding function (single sequence)
def DNA_to_onehot(seq, out_len = None):
    options_onehot = {
        'A': [1,0,0,0],
        'C': [0,1,0,0], 
        'G': [0,0,1,0],
        'T': [0,0,0,1],
        'N': [0,0,0,0],
        'K': [0,0,1,1],
        'M': [1,1,0,0],
        'R': [1,0,1,0],
        'Y': [0,1,0,1],
        'S': [0,1,1,0],
        'W': [1,0,0,1],
        'B': [0,1,1,1],
        'V': [1,1,1,0],
        'H': [1,1,0,1],
        'D': [1,0,1,1],
        'X': [1,1,1,1]
      }
    # check wheter we need to pad or truncante
    if out_len is not None and len(seq) > out_len :
        seq = seq[:out_len]
    if out_len is not None and len(seq) < out_len :
        seq = seq + 'N'*(out_len-len(seq))
    # cast to one-hot
    onehot_data = map(lambda e: options_onehot[e], seq.upper())
    return np.array(list(onehot_data))


# build one-hot encoding matrix (seveveral sequences)
def build_onehot_matrix(seqs, padd = True, truncate = False):
    # sanity check
    if truncate and padd:
        print("Error : cannot truncate and padd at the same time")
        return None
    # define output length
    seq_len = [len(x) for x in seqs]
    if truncate:
        out_len = np.min(seq_len)
    if padd:
        out_len = np.max(seq_len)
    # extract one-hot-encoding matrix
    one_hot_data = [DNA_to_onehot(x, out_len) for x in seqs]
    X = np.array(one_hot_data)
    # return
    return(X)


# build kmers to indices dictionary
def build_kmer_dic(seqs, k, debug = False):
    # initialize dictionary
    kmer_dic  = {}
    # build dictionary
    cptr = 0
    seq_cptr = 0
    debug = False
    for seq in seqs:
        seq_cptr = seq_cptr + 1
        for i in range(len(seq)-k+1):
            kmer = seq[i:i+k]
            if not bool(re.match('^[ATGC]+$', kmer)):
                if debug:
                    print("considering ATCG only kmers : skipping kmer {}".format(kmer))
                continue
            try:
                ind = kmer_dic[kmer]
            except:
                cptr = cptr + 1
                kmer_dic[kmer] = cptr
        # check all kmers are seen
        if len(kmer_dic) == 4**k:
                print("all possible {} k-mers are seen - stopping after sequence {} out of {}".format(4**k, seq_cptr + 1, len(seqs)))
                break
    print("found {} distinct k-mers out of {} possible".format(len(kmer_dic),4**k))
    return(kmer_dic)


# extract kmers tokens (single sequence)
def build_kmer_tokens_sequence(seq, k, kmer_dic, out_len = None):
    # check wheter we need to pad or truncante
    if out_len is not None and len(seq) > out_len :
        seq = seq[:out_len]
    if out_len is not None and len(seq) < out_len :
        seq = seq + 'N'*(out_len-len(seq))
    # extract sequences of indices
    kmers_ind = []
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        try: 
            ind = kmer_dic[kmer]
        except:
            ind = 0 # use index "0" if kmer not part of dictionary (e.g., "NNNN"'s for padding) 
        kmers_ind.append(ind)
    return(kmers_ind)


# build kmers tokens matrix (seveveral sequences)
def build_kmer_tokens_matrix(seqs, k, kmer_dic, padd = True, truncate = False):
    # sanity check
    if truncate and padd:
        print("Error : cannot truncate and padd at the same time")
        return None
    # define output length to consider
    seq_len = [len(x) for x in seqs]
    if truncate:
        out_len = np.min(seq_len)
    if padd:
        out_len = np.max(seq_len)
    # 
    seqs_token = [build_kmer_tokens_sequence(x, k, kmer_dic, out_len) for x in seqs]
    X = np.array(seqs_token)
    return(X)



# extract kmers profile (single sequence)
def build_kmer_profile_sequence(seq, k, kmer_dic):
    # initializer results
    n_km = len(seq) - k + 1
    km_prof = {}
    # evaluate kmers
    for i in range(n_km):
        # get  kmer
        kmer = seq[i:i + k]
        # get index
        try:
            index = kmer_dic[kmer]
            try:
                n_occ = km_prof[index]        # if kmer already seen : increment n.occ
                km_prof[index] = n_occ + 1
            except:                            # otherwise : add to profile
                km_prof[index] = 1
        except:
            index = 0
    # return
    return km_prof


# build kmers profile matrix (seveveral sequences)
def build_kmer_profile_matrix(seqs, k, kmer_dic, verbose = True):
    # monitor time
    start_time = time.time()
    # initialize results
    row_inds = []
    col_inds = []
    kmer_counts = []
    # process each sequence
    for seq_cptr, seq in enumerate(seqs):
        if verbose and (seq_cptr % 100 == 0):
            print("\t- kmerizing sequence {} out of {}".format(seq_cptr + 1, len(seqs)))
        # build kmer profile
        km_prof = build_kmer_profile_sequence(seq, k, kmer_dic)
        # get kmer counts and indices
        km_ind = np.fromiter(km_prof.keys(), dtype=int)
        km_counts = np.fromiter(km_prof.values(), dtype=int)
        # store
        row_inds.extend([seq_cptr] * len(km_ind))
        col_inds.extend(km_ind - 1)        # NB : kmer indices start at 1 in dictionary
        kmer_counts.extend(km_counts)
    # build sparse matrix
    X = coo_matrix((kmer_counts, (row_inds, col_inds)))
    # monitor time
    end_time = time.time()
    # show log
    if verbose:
        print("kmerization from {} sequences and k = {} took {} seconds".format(len(seqs), k, end_time - start_time))
    # cast to a standard numpy array and return
    Xd = X.toarray()
    return Xd


