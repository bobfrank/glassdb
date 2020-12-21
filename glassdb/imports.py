import sqlite3
import json
import csv
import time
import numpy as np
# TODO also support pq files?

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

from storage import create_table, insert_table_page, init_glassdb, TextVec

def add_page(fp_out, table_name, cols, rows, page_i):
    col_names = list(cols)
    table_page = dict((col.encode('utf-8'), [row.get(col).encode('utf-8') if isinstance(row.get(col),str) else row.get(col) for row in rows]) for col in col_names)
    for col in col_names:
        can_int = True
        can_float = True
        can_date = True
        is_list = False
        for r in range(min(len(rows),10)):
            if not try_int(rows[r][col]):
                can_int = False
            if not try_float(rows[r][col]):
                can_float = False
            if not try_date(rows[r][col]):
                can_date = False
            if isinstance(rows[r][col], list):
                is_list = True
        col = col.encode('utf-8')
        if can_int:
            table_page[col] = np.array(table_page[col], dtype=np.int64)
        elif can_float:
            table_page[col] = np.array(table_page[col], dtype=np.float64)
        elif can_date:
            table_page[col] = np.array(table_page[col], dtype=np.uint64)
        else:
            #print('res?',col, table_page[col])
            table_page[col] = TextVec.fromlist(table_page[col])
    insert_table_page(fp_out, table_name.encode('utf-8'), page_i, table_page, seed=1987)

def import_jsonl(fp_out, fp_in, table_name):
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
                create_table(fp_out, table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
            add_page(fp_out, table_name, cols, rows, page_i)
            page_i += 1
            rows[:] = []

    if len(rows) > 0:
        if page_i == 0:
            create_table(fp_out, table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
        add_page(fp_out, table_name, cols, rows, page_i)

def import_csv(fp_out, fp_in, table_name):
    reader = csv.reader(fp_in)
    header = next(reader)
    rows = list(reader)
    #return dict((col, [row[i] for row in rows]) for i,col in enumerate(header))

#        if len(rows) == 1000000:
#            if page_i == 0:
#                create_table(fp_out, table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
#            add_page(cols, rows, page_i)
#            page_i += 1
#            rows[:] = []
#
#    if len(rows) > 0:
#        if page_i == 0:
#            create_table(fp_out, table_name.encode('utf-8'), [col.encode('utf-8') for col in cols])
#        add_page(cols, rows, page_i)

def try_int(x):
    try:
        int(str(x))
        return True
    except:
        return False
def try_float(x):
    try:
        float(str(x))
        return True
    except:
        return False
def try_date(x):
    return False
def import_sqlite(fp, conn):
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

def main():
    with open('strat_returns.glass','wb+') as fp_out:
        init_glassdb(fp_out)
        with open('../tunnelvision/ls_strat_returns.jsonl') as fp_in:
            s = time.time()
            import_jsonl(fp_out, fp_in, 'strat_returns')
            print(time.time()-s)
    return
    with open('prices5s.glass','wb+') as fp:
        init_glassdb(fp)
        with sqlite3.connect('/home/bob/nas/data/finance/stocks/prices5s.db') as conn:
            s = time.time()
            res = import_sqlite(fp, conn)
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

