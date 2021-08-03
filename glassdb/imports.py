import sqlite3
import random
import json
import csv
import time
import numpy as np
import pyarrow.parquet as pq
from .storage import Storage, TextVec
from .table import TableFile


def jsonl_flatten(x, prefix=None):
    res = {}
    for k, v in x.items():
        if isinstance(v,list):
            pass # NOTE ignoring lists for now, probably should flatten to .0, .1, ...
        elif isinstance(v,dict):
            if prefix is None:
                res.update(jsonl_flatten(v, k))
            else:
                res.update(jsonl_flatten(v, '{}.{}'.format(prefix,k)))
        else:
            if prefix is None:
                res[k] = v
            else:
                res['{}.{}'.format(prefix,k)] = v
    return res

def asint(x):
    try:
        if isinstance(x, bytes):
            return int(x.replace(b',',b''))
        else:
            return int(str(x).replace(',',''))
    except:
        return np.nan

def asfloat(x):
    try:
        if isinstance(x, float):
            return x
        elif isinstance(x, bytes):
            return float(x.replace(b',',b''))
        else:
            return float(str(x).replace(',',''))
    except:
        return np.nan

def try_int(x):
    return not np.isnan(asint(x))

def try_float(x):
    if isinstance(x,float) and np.isnan(x):
        return True
    return not np.isnan(asfloat(x))

def try_date(x):
    return False

def add_page(tf, table_name, cols, rows, page_i):
    col_names = list(cols)
    if isinstance(rows, dict):
        table_page = rows
    else:
        table_page = dict((col.encode('utf-8'), [row.get(col).encode('utf-8') if isinstance(row.get(col),str) else row.get(col) for row in rows]) for col in col_names)
    seed = 1987
    gen = np.random.default_rng(seed=seed) # all db operations must be replicatable, so we need to pass a seed to build our sample for estimating percentiles
    for col in col_names:
        col = col.encode('utf-8')
        can_int = True
        can_float = True
        can_date = True
        is_list = False
        rs = gen.choice(range(len(table_page[col])), 10)
        for r in rs:
            if isinstance(table_page[col][r], (str,bytes)) and len(table_page[col][r]) == 0:
                continue
            if not try_int(table_page[col][r]):
                can_int = False
            if not try_float(table_page[col][r]):
                can_float = False
            if not try_date(table_page[col][r]):
                can_date = False
            if isinstance(table_page[col][r], list):
                is_list = True
        if can_int:
            table_page[col] = np.array(list(map(asint,table_page[col])), dtype=np.int64)
        elif can_float:
            table_page[col] = np.array(list(map(asfloat,table_page[col])), dtype=np.float64)
#        elif can_date:
#            table_page[col] = np.array(map(asdate,table_page[col]), dtype=np.uint64)
        else:
            table_page[col] = TextVec.fromlist(table_page[col])
    tf.insert_page(table_name.encode('utf-8'), page_i, table_page, seed=1987)

def import_jsonl(tf, fp_in, table_name):
    rows = []
    cols = set()
    page_i = 0
    for i,row in enumerate(fp_in):
        row_dict = jsonl_flatten(json.loads(row))
        if page_i == 0: # after page 0 cols can't change
            cols = cols.union(set(row_dict.keys()))
        rows.append(row_dict)

        if len(rows) == 1000000:
            if page_i == 0:
                tf.create_table(table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
            add_page(tf, table_name, cols, rows, page_i)
            page_i += 1
            rows[:] = []

    if len(rows) > 0:
        if page_i == 0:
            tf.create_table(table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
        add_page(tf, table_name, cols, rows, page_i)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def import_csv(tf, fp_in, table_name):
    reader = csv.reader(fp_in)
    header = next(reader)
    rows = list(reader)
    tf.create_table(table_name.encode('utf-8'), [col.encode('utf-8') for col in header])
    for page_i,page in enumerate(chunks(rows, 1000000)):
        page_data = dict((col.encode('utf-8'), [row[i].encode('utf-8') for row in rows]) for i,col in enumerate(header))
        add_page(tf, table_name, header, page_data, page_i)

def import_sqlite(fp, conn):
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        limit = 1000000
        for table, in list(cur.execute("select name from sqlite_master where type='table';")):
            #print(table)
            i = 0
            while True:
                #print(i*limit)
                cur.execute('SELECT * FROM {} LIMIT {} OFFSET {}'.format(table, limit, i*limit))
                header = [col[0].encode('utf-8') for col in cur.description]
                if i == 0:
                    create_table(fp, table.encode('utf-8'), header)
                rows = list(cur)
                if len(rows) == 0:
                    break
                table_page = dict((col, [row[k].encode('utf-8') if isinstance(row[k],str) else row[k] for row in rows]) for k,col in enumerate(header))
                for k, col in enumerate(header):
                    can_int = True
                    can_float = True
                    can_date = True
                    for r in range(min(len(rows),10)):
                        if not try_int(rows[r][k]):
                            can_int = False
                        if not try_float(rows[r][k]):
                            can_float = False
                        if not try_date(rows[r][k]):
                            can_date = False
                    if can_int:
                        table_page[col] = np.array(table_page[col], dtype=np.int64)
                    elif can_float:
                        table_page[col] = np.array(table_page[col], dtype=np.float64)
                    elif can_date:
                        table_page[col] = np.array(table_page[col], dtype=np.uint64)
                    else:
                        table_page[col] = TextVec.fromlist(table_page[col])
                insert_table_page(fp, table.encode('utf-8'), i, table_page, seed=1987)
                i += 1

def import_parquet(tf, path, table_name):
    table = pq.read_table(path)
    tf.create_table(table_name.encode('utf-8'), [col.encode('utf-8') for col in table.column_names])
    page_i = 0
    while True:
        table_page = {}
        for i,column_name in enumerate(table.column_names):
            col = table.columns[i]
            if page_i >= len(col.chunks):
                continue
            col_page = col.chunks[page_i]
            if col_page.type.equals('string'):
                texts = [x.encode('utf-8') for x in col_page.to_pylist()]
                table_page[column_name.encode('utf-8')] = TextVec.fromlist([x.encode('utf-8') for x in col_page.to_pylist()])
            # TODO elif col_page.type.equals('datetime'):
            else:
                table_page[column_name.encode('utf-8')] = col_page.to_numpy()
        if len(table_page) == 0:
            break
        tf.insert_page(table_name.encode('utf-8'), page_i, table_page, seed=1987)
        page_i += 1

def main():
    with open('strat_returns.glass','wb+') as fp_out:
        storage = Storage(fp_out)
        storage.init()
        tf = TableFile(storage)
        with open('../tunnelvision/ls_strat_returns.jsonl') as fp_in:
            s = time.time()
            import_jsonl(tf, fp_in, 'strat_returns')
            print(time.time()-s)
    return
    with open('prices5s.glass','wb+') as fp:
        storage = Storage(fp)
        storage.init()
        tf = TableFile(storage)
        with sqlite3.connect('/home/bob/nas/data/finance/stocks/prices5s.db') as conn:
            s = time.time()
            res = import_sqlite(tf, conn)
            print(time.time()-s)
            print(list(res.keys()))

    with open('/home/bob/nas/data/finance/etf.holdings.csv') as fp:
    #with open('../posight/EOD_metadata.csv') as fp:
        s = time.time()
        res = load_csv(fp)
        print(time.time()-s)
        #print(res)
        print(list(res.keys()))
        print(len(list(res.keys())), len(res[list(res.keys())[0]]))

if __name__ == '__main__':
    main()

