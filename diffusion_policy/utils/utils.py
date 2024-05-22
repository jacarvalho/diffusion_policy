def keep_idx_of_batch(dict_, idx=0):
    for k, v in dict_.items():
        dict_[k] = dict_[k][idx][None, ...]
