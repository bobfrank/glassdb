from .storage import mkdir, insert_vector, read_path, open_dir, TextVec, type_order, type_names, dir_max_entries
import numpy as np
import hashlib
import struct

def get_col_hashes(col):
    col_hash = int(hashlib.sha1(col).hexdigest(), 16)
    hash_dir_name = b'#'+str(col_hash%(dir_max_entries-2)).encode('utf-8')
    sub_hash_dir_name = b'#'+str(int(col_hash/(dir_max_entries-2))%(dir_max_entries-2)).encode('utf-8')
    return hash_dir_name, sub_hash_dir_name

def create_table(fp, table_name, col_names):
    table_offset_root,_ = mkdir(fp, 0, table_name)
    table_offset = table_offset_root
    hash_dir_offsets = {}
    sub_hash_dir_offsets = {}
    columns_offset = insert_vector(fp, table_offset, b'.columns', TextVec.fromlist(col_names))
    for col in col_names:
        hash_dir_name, sub_hash_dir_name = get_col_hashes(col)
        if hash_dir_name not in hash_dir_offsets:
            hash_dir_offset, table_offset = mkdir(fp, table_offset, hash_dir_name)
            hash_dir_offsets[hash_dir_name] = hash_dir_offset
            sub_hash_dir_offsets[hash_dir_name] = {}
        if sub_hash_dir_name not in sub_hash_dir_offsets[hash_dir_name]:
            sub_hash_dir_offset, hash_dir_offset = mkdir(fp, hash_dir_offsets[hash_dir_name], sub_hash_dir_name)
            sub_hash_dir_offsets[hash_dir_name][sub_hash_dir_name] = sub_hash_dir_offset
            hash_dir_offsets[hash_dir_name] = hash_dir_offset

        o, sub_hash_dir_offset = mkdir(fp, sub_hash_dir_offsets[hash_dir_name][sub_hash_dir_name], col)
        sub_hash_dir_offsets[hash_dir_name][sub_hash_dir_name] = sub_hash_dir_offset
    return table_offset_root

def insert_table_page(fp, table_name, page_index, table_page, seed=1987):
    gen = np.random.default_rng(seed=seed) # all db operations must be replicatable, so we need to pass a seed to build our sample for estimating percentiles
    table_offset = open_dir(fp, 0, table_name)
    hash_dir_offsets = {}
    sub_hash_dir_offsets = {}
    for col, vals in table_page.items():
        hash_dir_name, sub_hash_dir_name = get_col_hashes(col)
        if hash_dir_name not in hash_dir_offsets:
            hash_dir_offsets[hash_dir_name] = open_dir(fp, table_offset, hash_dir_name)
            sub_hash_dir_offsets[hash_dir_name] = {}
        if sub_hash_dir_name not in sub_hash_dir_offsets[hash_dir_name]:
            sub_hash_dir_offsets[hash_dir_name][sub_hash_dir_name] = open_dir(fp, hash_dir_offsets[hash_dir_name], sub_hash_dir_name)
        col_offset = open_dir(fp, sub_hash_dir_offsets[hash_dir_name][sub_hash_dir_name], col)
        page_offset, _ = mkdir(fp, col_offset, b'pg' + str(page_index).encode('utf-8'))
        values_offset = insert_vector(fp, page_offset, b'vals', vals)
        if len(vals) >= 100000:
            idx_offset, _ = mkdir(fp, page_offset, b'idx')
            values_sample = gen.choice(vals, 10000)
            if isinstance(vals, TextVec):
                sorted_values_sample = np.sort(values_sample)
                idx = np.array(np.arange(0,100)*len(values_sample)/100, dtype=np.uint32)
                percentile_estimates = TextVec.fromlist( sorted_values_sample[idx] )
            else:
                percentile_estimates = np.percentile(values_sample, np.arange(0,100))
            pct_offset = insert_vector(fp, idx_offset, b'pct', percentile_estimates)


def table_list_tables(fp):
    tables = []
    for name in read_path(fp, b'')['names'][1:]:
        if name.startswith(b'.'):
            continue
        tables.append(name)
    return tables

def table_list_columns(fp, table_name):
    return read_path(fp, '{}/.columns'.format(table_name.decode('utf-8')).encode('utf-8')).tolist()

def table_list_pages(fp, table_name):
    names = read_path(fp, table_name)['names']
    first_col_name = table_list_columns(fp, table_name)[0]
    hash_name, sub_hash_name = get_col_hashes(first_col_name)
    return read_path(fp, table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + first_col_name)['names']

def table_load_data(fp, table_name, col, page):
    hash_name, sub_hash_name = get_col_hashes(col)
    return read_path(fp, table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col + b'/' + page + b'/vals')

def table_list_columns_and_types(fp, table_name):
    result = []
    column_names = table_list_columns(fp, table_name)
    for col_name in column_names:
        hash_name, sub_hash_name = get_col_hashes(col_name)
        page0 = read_path(fp, table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col_name)['names'][0]
        page_rec = read_path(fp, table_name + b'/' + hash_name + b'/' + sub_hash_name + b'/' + col_name + b'/' + page0)
        type_idx = page_rec['types'][page_rec['names'].index(b'vals')]
        if isinstance(type_order[type_idx],str) and type_order[type_idx] != 'text':
            continue
        if col_name.startswith(b'.'):
            continue
        result.append((col_name, type_names[type_idx]))
    return result

#fd = os.open(lockfile, os.O_WRONLY|os.O_NOCTTY|os.O_CREAT, 0o666)
#if sys.platform.startswith('aix'):
#  # Python on AIX is compiled with LARGEFILE support, which changes the
#  # struct size.
#  op = struct.pack('hhIllqq', fcntl.F_WRLCK, 0, 0, 0, 0, 0, 0)
#else:
#  op = struct.pack('hhllhhl', fcntl.F_WRLCK, 0, 0, 0, 0, 0, 0)
#fcntl.fcntl(fd, fcntl.F_SETLK, op)

def main():
    from sql import _sql
    print(_sql("SELECT ticker,date,close FROM prices WHERE date>=:date"))
    s = time.time()
    with open('strat_returns.glass', 'rb') as fp:
        for page in read_path(fp, b'strat_returns/last_trade.SPY')['names']:
            a = read_path(fp, b'strat_returns/last_trade.SPY/%s/vals'%page)
            b = read_path(fp, b'strat_returns/returns.SPY/%s/vals'%page)
            print(a)
            #print(b)
    print(time.time()-s)
#    s = time.time()
#    with open('prices5s.glass', 'rb') as fp:
#        for page in read_path(fp, b'prices/open')['names']:
#            a = read_path(fp, b'prices/date/%s/idx/pct'%page)
#            b = read_path(fp, b'prices/close/%s/idx/pct'%page)
#            #b = read_path(fp, b'prices/open/%s/vals'%page)
#            #c = read_path(fp, b'prices/close/%s/vals'%page)
#            #print(a[0], a[-1])
#            print(a)
#    print(time.time()-s)
    return
    path = 'test.glass'
    vec1 = np.random.rand(1000000)
    vec2 = np.random.rand(1000000)
    #vals = np.array(200000*[3230,123981,.3498,.4,.99], dtype=np.float32)
    s = time.time()
    with open(path, 'wb+') as fp:
        init_glassdb(fp)
        create_table(fp, b'test-table', [b'col1',b'col2'])
        insert_table_page(fp, b'test-table', 0, {b'col1': vec1, b'col2': vec2})
    print(time.time()-s)
    s = time.time()

    with open(path, 'rb') as fp:
        col1 = read_path(fp, b'test-table/col1/pg0/vals')
        pct = read_path(fp, b'test-table/col1/pg0/idx/pct')
    print(time.time()-s)
    print(col1)
    print(len(col1))
    #print(pct)

if __name__ == '__main__':
    main()


