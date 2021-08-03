from .storage import TextVec, type_order, type_names, Directory
import numpy as np
import hashlib
import struct
import time
import bisect

def get_col_hashes(col):
    col_hash = int(hashlib.sha1(col).hexdigest(), 16)
    hash_dir_name = b'hash'+str(col_hash%(Directory.dir_max_entries-2)).encode('utf-8')
    sub_hash_dir_name = b'hash'+str(int(col_hash/(Directory.dir_max_entries-2))%(Directory.dir_max_entries-2)).encode('utf-8')
    return hash_dir_name, sub_hash_dir_name

def crack_to_pivot_value(pivot_value, ls, partitions, partition_offsets, idxs):
    # this is basically the offset into the indexes for a given partition's bound
    changed = False
    if partition_offsets is None:
        partition_offsets = np.array([-1]*len(partitions), dtype=np.int32)
        changed = True
        needs_copy = False
    else:
        needs_copy = True

    if idxs is None:
        idxs = np.arange(0, len(ls), dtype=np.uint32)
        changed = True
    else:
        needs_copy = True

    # figure out what partition I'm interested in for this pivot value
    if isinstance(ls, TextVec):
        right_pct_idx = bisect.bisect_left(partitions, pivot_value)
    else:
        right_pct_idx = np.searchsorted(partitions, pivot_value, side='left')
    left_pct_idx = right_pct_idx-1

    # calculate the sequence of bounds to get to this point
    work = []
    k = 1
    while True:
        work.append((left_pct_idx, right_pct_idx))
        if 2**(k+1) >= len(partitions):
            break
        left_pct_idx = int(left_pct_idx/2**k)*2**k
        right_pct_idx = int(right_pct_idx/2**k+1)*2**k
        k += 1

    # finally apply the cracking
    lo = 0
    ro = len(idxs)
    cracked = False
    for left, right in reversed(work):
        if left >= 0:
            if partition_offsets[left] == -1:
                below_left = ls[idxs[lo:ro]] <= partitions[left]
                idxs0 = idxs[lo:ro][below_left]
                idxs1 = idxs[lo:ro][~below_left]
                if needs_copy:
                    idxs = np.array(idxs)
                    partition_offsets = np.array(partition_offsets)
                    needs_copy = False
                idxs[lo:lo+len(idxs0)] = idxs0
                idxs[lo+len(idxs0):ro] = idxs1
                partition_offsets[left] = lo + len(idxs0)
                cracked = True
            lo = partition_offsets[left]
        else:
            lo = 0

        if right < len(partition_offsets):
            if partition_offsets[right] == -1:
                below_right = ls[idxs[lo:ro]] <= partitions[right]
                idxs0 = idxs[lo:ro][below_right]
                idxs1 = idxs[lo:ro][~below_right]
                if needs_copy:
                    idxs = np.array(idxs)
                    partition_offsets = np.array(partition_offsets)
                    needs_copy = False
                idxs[lo:lo+len(idxs0)] = idxs0
                idxs[lo+len(idxs0):ro] = idxs1
                partition_offsets[right] = lo + len(idxs0)
                cracked = True
            ro = partition_offsets[right]
        else:
            ro = len(idxs)

    # always sort the partition of interest (for now, could be a flag eventually since its sometimes causing applying the indexes to be slower for filters)
    # TODO /idx/partition-sorted-flags or.. just always sort once down to the single buckets
    if cracked:
        idxs_hat = idxs[lo:ro]
        ls_idx_idx = np.argsort(ls[idxs_hat])
        idxs[lo:ro] = idxs_hat[ls_idx_idx]
        changed = True

    # TODO pivoting by < vs <=
    idxs_hat = idxs[lo:ro]
    if isinstance(ls, TextVec):
        pivot_idx_idx = lo + bisect.bisect_left(ls[idxs_hat], pivot_value)
    else:
        pivot_idx_idx = lo + np.searchsorted(ls[idxs_hat], pivot_value, side='left')

    return changed, partition_offsets, idxs, pivot_idx_idx


class TableFile(object):
    def __init__(self, storage):
        self.storage = storage

    def create_table(self, table_name, col_names):
        self.storage.insert_vector(table_name, b'.columns', TextVec.fromlist(col_names))

    def crack_column_to_pivot_value(self, table_name, col, page, ls, pivot_value, disable_compression=False):
        hash_name, sub_hash_name = get_col_hashes(col)
        idx_path = table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col + b'/' + page + b'/idx'

        partitions = self.storage.read_path(idx_path + b'/pct')
        partition_offsets = self.storage.read_path(idx_path + b'/pct-offsets')
        indexes = self.storage.read_path(idx_path + b'/indexes')
        if partition_offsets is None:
            insert_vectors = True
        else:
            insert_vectors = False

        changed, partition_offsets, indexes, pivot_idx_idx = crack_to_pivot_value(pivot_value, ls, partitions, partition_offsets, indexes)

        if changed:
            if insert_vectors:
                self.storage.insert_vector(idx_path, b'pct-offsets', partition_offsets, disable_compression=disable_compression)
                self.storage.insert_vector(idx_path, b'indexes', indexes, disable_compression=disable_compression)
            else:
                self.storage.update_vector(idx_path, b'pct-offsets', partition_offsets, disable_compression=disable_compression)
                self.storage.update_vector(idx_path, b'indexes', indexes, disable_compression=disable_compression)

        return indexes, pivot_idx_idx

    # not really cracking a full index, once you get to the 5-10k row range its about the same time vs a proper index
    # TODO need to think about the sorting aspect of cracking
    # global index could be two uint32s, one for page number, one for index within the page..
    # updates to data will change the index, especially the global indexes... need to think about that TODO


    def insert_page(self, table_name, page_index, table_page, seed=1987, disable_compression=False, disable_stats=None):
        gen = np.random.default_rng(seed=seed) # all db operations must be replicatable, so we need to pass a seed to build our sample for estimating percentiles
        for col, vals in table_page.items():
            hash_dir_name, sub_hash_dir_name = get_col_hashes(col)
            page_path = table_name + b'/' + hash_dir_name + b'/' + sub_hash_dir_name + b'/' + col + b'/pg' + str(page_index).encode('utf-8')
            values_offset = self.storage.insert_vector(page_path, b'vals', vals, disable_compression=disable_compression)

            # build sample stats
            if len(vals) >= 2**14 and not disable_stats:
                nsamples = int(len(vals)/2**13)
                idx_path = page_path + b'/idx'
                values_sample = gen.choice(vals, 10000)
                if isinstance(vals, TextVec):
                    sorted_values_sample = np.sort(values_sample)
                    idx = np.array(np.arange(0,nsamples)*len(values_sample)/nsamples, dtype=np.uint32)
                    percentile_estimates = TextVec.fromlist( sorted_values_sample[idx] )
                else:
                    percentile_estimates = np.percentile(values_sample, np.arange(0,nsamples)*100/nsamples)
                pct_offset = self.storage.insert_vector(idx_path, b'pct', percentile_estimates, disable_compression=disable_compression)


    def list_tables(self):
        tables = []
        for name in self.storage.read_path(b'').names[1:]:
            if name.startswith(b'.'):
                continue
            tables.append(name)
        return tables

    def list_columns(self, table_name):
        return self.storage.read_path('{}/.columns'.format(table_name.decode('utf-8')).encode('utf-8')).tolist()

    def list_pages(self, table_name):
        first_col_name = self.list_columns(table_name)[0]
        hash_name, sub_hash_name = get_col_hashes(first_col_name)
        path = table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + first_col_name
        rec = self.storage.read_path(path)
        if rec is None:
            return []
        else:
            return rec.names

    def load_data(self, table_name, col, page):
        hash_name, sub_hash_name = get_col_hashes(col)
        path = table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col + b'/' + page + b'/vals'
        return self.storage.read_path(path)

    def update_data(self, table_name, col, page, new_vals, disable_compression=False):
        hash_name, sub_hash_name = get_col_hashes(col)
        path = table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col + b'/' + page
        self.storage.update_vector(path, b'vals', new_vals, disable_compression=disable_compression)

    def list_columns_and_types(self, table_name):
        result = []
        column_names = self.list_columns(table_name)
        for col_name in column_names:
            hash_name, sub_hash_name = get_col_hashes(col_name)
            page0 = self.storage.read_path(table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col_name).names[0]
            page_rec = self.storage.read_path(table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col_name + b'/' + page0)
            type_idx = page_rec.types[page_rec.names.index(b'vals')]
            if isinstance(type_order[type_idx],str) and type_order[type_idx] != 'text':
                continue
            if col_name.startswith(b'.'):
                continue
            result.append((col_name, type_names[type_idx]))
        return result

