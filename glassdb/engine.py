import numpy as np
from sql import _sql
from storage import read_path
import time
import copy

# TODO joins are complicated
# TODO all these encode('utf-8')'s need to be cleaned up (into one place), also make sure unicode all works well inside sql too

def explode_conditions(cond, path):
    result = []
    if isinstance(cond, tuple) and cond[0] in ['and','or']:
        result.extend(explode_conditions(cond[1], path+[1]))
        result.extend(explode_conditions(cond[2], path+[2]))
    elif len(cond):
        result.append((path, cond))
    return result

def replace_tuple(cond, path, lr, value):
    if len(path) == 0:
        return (cond[0], value, cond[2]) if lr == 'left' else (cond[0], cond[1], value)
    else:
        nv = replace_tuple(cond[path[0]], path[1:], lr, value)
        return tuple(nv if path[0] == i else cond[i] for i in range(len(cond)))

def evaluate_conditions(conditions):
    if len(conditions) == 0:
        return True
    elif conditions[0] == 'and':
        return evaluate_conditions(conditions[1]) & evaluate_conditions(conditions[2])
    elif conditions[0] == 'or':
        return evaluate_conditions(conditions[1]) | evaluate_conditions(conditions[2])
    elif conditions[0] == '<=':
        return conditions[1] <= conditions[2]
    elif conditions[0] == '<':
        return conditions[1] < conditions[2]
    elif conditions[0] == '>=':
        return conditions[1] >= conditions[2]
    elif conditions[0] == '>':
        return conditions[1] > conditions[2]
    elif conditions[0] == '==':
        return conditions[1] == conditions[2]
    elif conditions[0] == '!=':
        return conditions[1] > conditions[2]
    else:
        raise NotImplementedError('havent implemented condition {}'.format(conditions[0]))

def execute_query_plan(fp, table):
    table_name = table['from'].encode('utf-8')

    select = table['select']
    if '*' in select:
        idx = select.index('*')
        select.pop(idx)
        all_cols = [col.decode('utf-8') for col in read_path(fp, b'%s'%(table_name))['names']]
        for col in reversed(all_cols):
            select.insert(idx, col)
    conditions = table.get('where',[])
    ex_conditions = explode_conditions(conditions, [])
    by_col = {}
    for path, cond in ex_conditions:
        left = cond[1]
        right = cond[2]
        if isinstance(left, (str,tuple)):
            if isinstance(left,tuple) and left[0] == '.':
                left = '.'.join(left[1:]).encode('utf-8')
            else:
                left = left.encode('utf-8')
            by_col.setdefault(left, []).append((path, 'left'))
        if isinstance(right, (str,tuple)):
            if isinstance(right,tuple) and right[0] == '.':
                right = '.'.join(right[1:]).encode('utf-8')
            else:
                right = right.encode('utf-8')
            by_col.setdefault(right, []).append((path, 'right'))
    def encode_order(x):
        if isinstance(x,tuple) and x[-1].startswith('!'):
            return encode_order(x[0])
        elif isinstance(x,tuple):
            return '.'.join(x[1:]).encode('utf-8')
        else:
            return x.encode('utf-8')
    used_columns = set('.'.join(x[1:]).encode('utf-8') if isinstance(x,tuple) else x.encode('utf-8') for x in select).union(by_col.keys()).union([encode_order(x) for x in table.get('order',[])])
    result = {}
    if len(used_columns):
        col0 = list(used_columns)[0]
        pages = read_path(fp, b'%s/%s'%(table_name,col0))['names']
        for page in pages:
            page_cond = copy.deepcopy(conditions)
            page_vectors = {}
            for column_name in used_columns:
                # any filters and orderings that can be evaluated now.. should be
                # should also be triggering cracking as needed
                page_vec = read_path(fp, b'%s/%s/%s/vals'%(table_name, column_name, page))
                page_vectors[column_name] = page_vec

                col_cond = by_col.get(column_name,[])
                for path, lr in col_cond:
                    page_cond = replace_tuple(page_cond, path, lr, page_vec)
            page_filter = evaluate_conditions(page_cond)
            for col in select:
                if isinstance(col, tuple):
                    col_name = '.'.join(col[1:]).encode('utf-8')
                else:
                    col_name = col.encode('utf-8')
                if isinstance(page_filter,bool) and page_filter == True:
                    result.setdefault(page,{})[col_name] = page_vectors[col_name]
                else:
                    result.setdefault(page,{})[col_name] = page_vectors[col_name][page_filter]

            if 'order' in table:
                order = table['order']
                if len(pages) != 1:
                    raise RuntimeError('multi-page sorting is not implemented yet')
                def extract_order_vec(x):
                    if isinstance(x,tuple) and x[-1] == '!asc':
                        return extract_order_vec(x[0])
                    elif isinstance(x,tuple) and x[-1] == '!desc':
                        return -extract_order_vec(x[0])
                    else:
                        return page_vectors['.'.join(x[1:]).encode('utf-8') if isinstance(x,tuple) else x.encode('utf-8')]
                idx = np.lexsort( tuple(extract_order_vec(x) for x in reversed(order)) )
                for col in result[page]:
                    result[page][col] = result[page][col][idx]
    if 'limit' in table:
        limit = table['limit']
        for page in list(sorted(result.keys())):
            if limit == 0:
                del result[page]
            else:
                for col in result[page]:
                    result[page][col] = result[page][col][:limit]
                limit = max(len(result[page][list(result[page])[0]]) - limit, 0)
    # anything that couldn't be taken above should be evaluated now
    return result

# ('order', ['date'], ('where', ('>', ('.', 'returns', 'SPY'), 0), ('from', 'strat_returns', ('select', ['date', ('.', 'last_trade', 'SPY'), ('.', 'returns', 'SPY')]))))

def translate_parse_tree(fp, query_tuple):
    # TODO should be reading the file to figure out what indexes exist / what should be cracked and what should be sorted/filtered on the fly
    # TODO also the result shouldn't be a 'table' with operations but a full query plan that can be executed without much thought from the final execution engine
    if len(query_tuple) == 3:
        table = translate_parse_tree(fp, query_tuple[2])
    else:
        table = {}
    for key in ['select','from','where','order','group','having','limit']:
        if query_tuple[0] == key:
            table[key] = query_tuple[1]
    return table

def replace_parse_tree_kwargs(query_tuple, kwargs):
    if isinstance(query_tuple, tuple):
        if query_tuple[0] == ':':
            return kwargs[ query_tuple[1] ]
        else:
            return tuple(replace_parse_tree_kwargs(val, kwargs) for val in query_tuple)
    else:
        return query_tuple

def execute(fp, query_tuple, kwargs):
    query_tuple = replace_parse_tree_kwargs(query_tuple, kwargs)
    table = translate_parse_tree(fp, query_tuple)
    return execute_query_plan(fp, table)

# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', ['test', '*'])))))
# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', [('.', 'test', '*')])))))
# ('where', ('and', ('like', 'id', (':',)), ('>=', 'date', 20201104)), ('from', 'symbols', ('select', [('count', '*')])))
# ('where', ('like', 'id', (':',)), ('from', 'symbols', ('select', [('count', '*')])))

if __name__ == '__main__':
    s = time.time()
    query = _sql("SELECT strategy.name, date, last_trade.SPY, returns.SPY, returns.XLE, returns.XLK FROM strat_returns WHERE strategy.name='up' AND returns.SPY>:minret ORDER BY date")
    with open('strat_returns.glass', 'rb') as fp:
        res = execute(fp, query, {'minret': 0.005})
    print(time.time()-s)
    print(res)
    print(len(res[b'\0\0\0\0pg'][b'date']))

# TODO expose (1) as a library (to be used like sqlite or pandas), (2) as a server (to be used like mssql or clickhouse), and (3) as a command line interface to use sql queries on arbitrary files that can be imported

