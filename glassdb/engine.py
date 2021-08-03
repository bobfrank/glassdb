import numpy as np
from .sql import _sql
from .storage import Storage, TextVec, DatetimeVec
from .table import TableFile, get_col_hashes
import time
import copy
import string
import os

# TODO all these encode('utf-8')'s need to be cleaned up (into one place), also make sure unicode all works well inside sql too
def table_evaluate_condition(op, lhs, rhs):
    vals = lhs[0]
    if len(vals) < 20000: # just directly compare these
        if op == '<=':
            return lhs[0] <= rhs[0]
        elif op == '<':
            return lhs[0] < rhs[0]
        elif op == '>=':
            return lhs[0] >= rhs[0]
        elif op == '>':
            return lhs[0] > rhs[0]
        else:
            raise RuntimeError(f'table conditions only handle <=, <, >=, >, not {op}')
    else:
        tf, table_name, col, page = lhs[1]
        idxs, pivot_idx_idx = tf.crack_column_to_pivot_value(table_name, col, page, lhs[0], rhs)
        # TODO <= vs <, and maybe add equality and inequality here.. could be faster too
        if op in ['<=','<']:
            s = time.time()
            res = idxs[:pivot_idx_idx]
            print('filter',time.time()-s)
            return res
        elif op in ['>=','>']:
            s = time.time()
            res = idxs[pivot_idx_idx:]
            print('filter',time.time()-s)
            return res
        else:
            raise RuntimeError('unhandled crack op: %s'%op)

# TODO could make sense to only evaluate the condition after filtering the rhs based on the lhs's results (or the other way could be better too.. could be worth figuring that out from the pct data)
# from egregium.py:
#def intersect(metrics, bounds):
#    # probably can do this much faster with cracked indexes in glassdb..
#    res = None
#    for (left, right), metric in zip(bounds, metrics):
#        if not np.isnan(left):
#            if res is None:
#                res = (metric>left)
#            else:
#                res &= (metric>left)
#        if not np.isnan(right):
#            if res is None:
#                res = (metric<right)
#            else:
#                res &= (metric<right)
#    #res_prev = np.roll(res,1)
#    #res_prev[0] = False
#    #res[res_prev == res] = False
#    return res

def evaluate_conditions(conditions, filter_idx=None):
    if len(conditions) == 0:
        return False, True
    elif conditions[0] == 'and':
        is_idx, lhs = evaluate_conditions(conditions[1], filter_idx=filter_idx)
        if is_idx:
            return True, evaluate_conditions(conditions[2], filter_idx=lhs)[1]
        else:
            is_idx, rhs = evaluate_conditions(conditions[2], filter_idx=filter_idx)
            if is_idx:
                raise RuntimeError('unhandled for now')
            else:
                return False, lhs & rhs
    elif conditions[0] == 'or':
        is_idx, lhs = evaluate_conditions(conditions[1], filter_idx=filter_idx)
        if is_idx:
            raise RuntimeError('unhandled for now')
        else:
            is_idx, rhs = evaluate_conditions(conditions[2], filter_idx=filter_idx)
            if is_idx:
                raise RuntimeError('unhandled for now')
            else:
                return False, lhs | rhs
    elif conditions[0] in ['<=','<','>=','>'] and (isinstance(conditions[1], tuple) or isinstance(conditions[2], tuple)):
        idxs = table_evaluate_condition(conditions[0], conditions[1], conditions[2])
        if filter_idx is None:
            return True, idxs
        else:
            return True, np.intersect1d(idxs, filter_idx, assume_unique=True)
    elif conditions[0] == '<=':
        if filter_idx is None:
            return False, conditions[1] <= conditions[2]
        else:
            return True, filter_idx[conditions[1][filter_idx] <= conditions[2]]
    elif conditions[0] == '<':
        if filter_idx is None:
            return False, conditions[1] < conditions[2]
        else:
            return True, filter_idx[conditions[1][filter_idx] < conditions[2]]
    elif conditions[0] == '>=':
        if filter_idx is None:
            return False, conditions[1] >= conditions[2]
        else:
            return True, filter_idx[conditions[1][filter_idx] >= conditions[2]]
    elif conditions[0] == '>':
        if filter_idx is None:
            return False, conditions[1] > conditions[2]
        else:
            return True, filter_idx[conditions[1][filter_idx] > conditions[2]]
    elif conditions[0] == '==':
        if isinstance(conditions[1],tuple):
            ls = conditions[1][0]
        else:
            ls = conditions[1]
        if filter_idx is None:
            return False, ls == conditions[2]
        else:
            return True, filter_idx[ls[filter_idx] == conditions[2]]
    elif conditions[0] == '!=':
        if isinstance(conditions[1],tuple):
            ls = conditions[1][0]
        else:
            ls = conditions[1]
        if filter_idx is None:
            return False, ~(ls == conditions[2])
        else:
            return True, filter_idx[~(ls[filter_idx] == conditions[2])]
    elif conditions[0] == 'in':
        if filter_idx is None:
            if isinstance(conditions[1], TextVec):
                return False, conditions[1].isin(conditions[2])
            else:
                return False, np.isin(conditions[1], conditions[2])
        else:
            if isinstance(conditions[1],tuple):
                ls = conditions[1][0]
            else:
                ls = conditions[1]
            if isinstance(conditions[1], TextVec):
                return False, ls[filter_idx].isin(conditions[2])
            else:
                return False, np.isin(ls[filter_idx], conditions[2])
    else:
        raise NotImplementedError('havent implemented condition {}'.format(conditions[0]))

def extract_order_vec(x, values):
    if isinstance(x,tuple) and x[-1] == '!asc':
        return extract_order_vec(x[0], values)
    elif isinstance(x,tuple) and x[-1] == '!desc':
        return -extract_order_vec(x[0], values)
    else:
        return values['.'.join(x[1:]) if isinstance(x,tuple) else x]

def execute_plan(cb, tf, staging_tf, plan):
    ip = 0
    values = {}
    source = {}
    s = time.time()
    while ip < len(plan):
        cmd = plan[ip]
        print('cmd',cmd)
        s = time.time()
        result_name = cmd[0]
        method = cmd[1]
        args = []
        clear_source = True

        for arg in cmd[2:]:
            if isinstance(arg,str):
                variables = [fn for _, fn, _, _ in string.Formatter().parse(arg) if fn is not None]
                kw = {}
                for var in variables:
                    kw[var] = values[var]['items'][ values[var]['index'] ].decode('utf-8')
                args.append(arg.format(**kw).encode('utf-8'))
            else:
                args.append(arg)

        if method == 'step':
            list_name = args[0].decode('utf-8')
            values[list_name]['index'] += 1
            if values[list_name]['index'] < len(values[list_name]['items']):
                ip = values[args[1].decode('utf-8')] # going to be incremented before ip is executed.. but points to 'label' command, so this will end up skipping that op

        elif method == 'label':
            values[result_name] = ip
            #print(result_name, ip)

        elif method == 'list-pages':
            table_name = args[0]
            if table_name == b':stage:':
                names = staging_tf.list_pages(table_name)
            else:
                names = tf.list_pages(table_name)
            values[result_name] = {'items': names, 'index': 0}
            if len(names) == 0:
                # just jump to step... TODO need to have this described in the plan
                while ip+1 < len(plan) and plan[ip+1][1] != 'step':
                    ip += 1

        elif method == 'load-table-data':
            #plan.append( (col, 'load-table-data', scope['__table__'], col, f'{scope["__table__"]}_pages') )
            table_name = args[0]
            col = args[1]
            page = args[2]
            if table_name == b':stage:':
                values[result_name] = staging_tf.load_data(table_name, col, page)
                source[result_name] = (staging_tf, table_name, col, page)
                clear_source = False
            else:
                values[result_name] = tf.load_data(table_name, col, page)
                source[result_name] = (tf, table_name, col, page)
                clear_source = False

        elif method == 'eval-condition':
            # TODO cracking if these are directly from pages in the table
            enable_cracking = True
            def replace_values(cond):
                if isinstance(cond, tuple) and cond[0] == '.' and '.'.join(cond[1:]) in values:
                    cond_hat = '.'.join(cond[1:])
                    if cond_hat in source and enable_cracking:
                        return (values[cond_hat],source[cond_hat])
                    else:
                        return values[cond_hat]
                elif isinstance(cond, str) and cond in values:
                    if cond in source and enable_cracking:
                        return (values[cond],source[cond])
                    else:
                        return values[cond]
                elif isinstance(cond, tuple):
                    return tuple(replace_values(x) for x in cond)
                else:
                    return cond

            is_idx, res = evaluate_conditions(replace_values(cmd[2]))
            if is_idx:
                values[result_name] = res
            else:
                # marginal improvement by changing from boolean into list of indexes when final list is pretty small compared to total list for this page
                idxs = np.arange(0, len(res))[res]
                if len(idxs) < 0.5*len(res):
                    values[result_name] = idxs
                else:
                    values[result_name] = res

        elif method == 'apply-filter':
            values[result_name] = values[cmd[2]][values[cmd[3]]]

        elif method == 'build-sort-index':
            order = cmd[2]
            values[result_name] = np.lexsort( tuple(extract_order_vec(x, values) for x in reversed(order)) )

        elif method == 'sort-to-stage':
            columns = cmd[2]
            order = cmd[3]
            stage_table_name = b':stage:'

            # we are first.. initialize glassdb setup
            off = os.lseek(staging_tf.storage.fp.fileno(), 0, os.SEEK_CUR)
            #print('off',off)
            if off == 0:
                staging_tf.storage.init()

            # create the staging if it doesn't exist yet TODO needs to be indexed by column list
            tables = staging_tf.list_tables()
            #print('tables',tables, stage_table_name)
            if stage_table_name not in tables:
                #print('creating..')
                staging_tf.create_table(stage_table_name, [col.encode('utf-8') for col in columns])

            #print('staging-file',staging_tf.fileno())
            staged_pages = staging_tf.list_pages(stage_table_name)
            #print('len(tickers)',len(values.get('ticker',[])), values.get('ticker'))
            if len(staged_pages) == 0:
                # first page.. just add what we have
                table_page = dict((k.encode('utf-8'),values[k]) for k in columns)
                #print(table_page)
                staging_tf.insert_page(stage_table_name, 0, table_page, disable_compression=True, disable_stats=True)
            else:
                # other pages.. load each page, do a 2-way merge, and save back the first page, then at the end save back the last page
                tmp_values = {}
                for col in columns:
                    tmp_values[col.encode('utf-8')] = values[col]
                for page in staged_pages:
                    merged_values = {}
                    count = 0
                    for col in columns:
                        col_vals = staging_tf.load_data(stage_table_name, col.encode('utf-8'), page)
                        if isinstance(col_vals, TextVec):
                            merged_values[col] = TextVec.concatenate((col_vals, tmp_values[col.encode('utf-8')]))
                        else:
                            merged_values[col] = np.concatenate((col_vals, tmp_values[col.encode('utf-8')]))
                        count = len(col_vals)
                    # TODO perhaps we can bucket these and sort the buckets independently instead.. could be slower.. but might be faster, needs a test to decide
                    merged_index = np.lexsort( tuple(extract_order_vec(x, merged_values) for x in reversed(order)) )
                    page_0 = merged_index[:count]
                    page_1 = merged_index[count:]
                    for col in columns:
                        staging_tf.update_data(stage_table_name, col.encode('utf-8'), page, merged_values[col][page_0], disable_compression=True)
                        tmp_values[col.encode('utf-8')] = merged_values[col][page_1]
                #print(tmp_values)
                staging_tf.insert_page(stage_table_name, len(staged_pages), tmp_values, disable_compression=True, disable_stats=True)

            #plan.append((None,'sort-to-stage',args, scope['__sort__']))
            # (None, 'sort-to-stage', ['ticker', 'date'], [('.', 'ticker')]),
    
        elif method == 'build-group-indexes':
            group_by = cmd[2]
            if len(group_by) != 1:
                raise NotImplementedError('Only can handle group by for one parameter for now')
            for col in group_by:
                if isinstance(col, tuple):
                    col = '.'.join(col[1:])
                if isinstance(values[col], TextVec):
                    projection = values[col].groupby_order()
                    gbo = values[col][projection]
                    start_flag = gbo.groupby_starts()
                    start_indexes = np.concatenate(([0], np.flatnonzero(start_flag)+1))
                    #print('start_flag', start_flag)
                    #print('start_indexes', start_indexes)
                    values[result_name] = {'project': projection, 'start_indexes': start_indexes}
                else:
                    projection = np.argsort(values[col])
                    gbo = values[col][projection]
                    vhat = np.roll(gbo,1)
                    vhat[0] = -1
                    start_flag = vhat != gbo
                    start_indexes = np.concatenate(([0], np.flatnonzero(start_flag)+1))
                    values[result_name] = {'project': projection, 'start_indexes': start_indexes}

        elif method == 'group-reduce-to-stage':
            group_by = cmd[2]
            agg_func = cmd[3]
            agg_col = cmd[4]
            final_name = '{}({})'.format(agg_func, agg_col)
            agg_value = values[ agg_col ]
            agg_value_ordered = agg_value[ values[group_by]['project'] ]
            start_indexes = values[group_by]['start_indexes']
            print(list(agg_value[start_indexes]))
            if agg_func == 'first':
                result = agg_value_ordered[start_indexes]
            elif agg_func == 'sum':
                result = np.add.reduceat(agg_value_ordered, start_indexes)
            else:
                raise NotImplementedError('havent implemented fn `{}` yet'.format(agg_func))
            #values.setdefault('__staged__',{})[final_name] = result TODO

# TODO joins should be supported in a similar way to the sorting (e.g. add a table into the staging table then merge the next table in by join key and then the next table, etc

        elif method == 'push-limited':
            values.setdefault('push-limited::used',0)
            used = values['push-limited::used']
            limit = cmd[3] - used
            if not isinstance(cmd[2], list) and  cmd[2] in values:
                res = values[cmd[2]][:limit]
                values['push-limited::used'] += len(res)
                cb( res )
                if len(values[cmd[2]]) >= limit:
                    ip += 1
            else:
                count = len(values[list(cmd[2])[0]][:limit])
                values['push-limited::used'] += count
                cb( dict((var,values[var][:limit]) for var in cmd[2]) )
                if len(values[cmd[2][0]]) >= limit:
                    ip += 1

        elif method == 'push':
            if not isinstance(cmd[2], list) and  cmd[2] in values:
                cb( values[cmd[2]] )
            else:
                cb( dict((var,values[var]) for var in cmd[2]) )

        ip += 1
        if clear_source and result_name in source:
            del source[result_name]
        print('   ',time.time()-s)

def build_query_plan(plan, tf, parse_tree, scope=None):
    if scope is None:
        scope = {}

    def labelize(x):
        return '.'.join(x[1:]) if isinstance(x,tuple) and x[0] == '.' else x

    print('parse_tree',parse_tree)
    if parse_tree[0] == 'select':
        res = []
        if '__table__' in scope:
            filter_cols = []
            def extract_filter_columns(tree):
                if isinstance(tree, tuple) and tree[0] == '.':
                    filter_cols.append('.'.join(tree[1:]))
                elif isinstance(tree, (list,tuple)):
                    for tree_k in tree:
                        extract_filter_columns(tree_k)
            if '__where__' in scope:
                extract_filter_columns(scope['__where__'])
            if '__group__' in scope:
                group_cols = scope['__group__']
            else:
                group_cols = []
            if '__order__' in scope:
                order_cols = scope['__order__']
            else:
                order_cols = []
            projection = list(map(labelize,parse_tree[1]))
            columns = [col.decode('utf-8') for col in tf.list_columns(scope['__table__'].encode('utf-8'))]
            if '*' in projection or '{}.*'.format(scope['__table__']) in projection:
                print('columns',columns)
                projection = [p for p in projection if p != '*' and p != '{}.*'.format(scope['__table__'])] + list(map(labelize,columns))
            print(order_cols, group_cols, filter_cols)
            cols = set(projection).union(map(labelize,filter_cols)).union(map(labelize,group_cols)).union(map(labelize,order_cols))
#            for col in projection:
#                if col in columns and col in cols:
#                    res.append(col)
            for col in cols:
                if col in columns:
                    plan.append( (col, 'load-table-data', scope['__table__'], col, '{%s_pages}'%scope['__table__']) )
                    res.append(col)
                else:
                    raise RuntimeError('Unable to find column %s'%col)
        return {'available': res, 'projection': projection}

    elif parse_tree[0] == 'from':
        table_name = parse_tree[1]
        plan.append( ('{}_pages'.format(table_name), 'list-pages', table_name) )
        plan.append( ('{}_start'.format(table_name), 'label') )
        result = build_query_plan(plan, tf, parse_tree[2], scope=dict(scope,__table__=table_name))

        if '__where__' in scope:
            plan.append(('idx','eval-condition',scope['__where__']))
            for arg in result['available']:
                plan.append((arg, 'apply-filter',arg,'idx'))

        staged = False
        if '__group__' in scope:
            plan.append(('__group__', 'build-group-indexes', scope['__group__']))
            for arg in result['available']:
                plan.append((None, 'group-reduce-to-stage', '__group__', scope['__agg__'].get(arg,'first'), arg))
            staged = True
        elif '__sort__' in scope:
            # TODO order by + group by?
            plan.append(('__order__', 'build-sort-index', scope['__sort__']))
            for arg in result['available']:
                plan.append((arg, 'apply-filter', arg, '__order__'))
            plan.append((None,'sort-to-stage', result['available'], scope['__sort__']))
            staged = True
        elif '__limit__' in scope:
            plan.append((None,'push-limited',result['projection'], scope['__limit__']))
        else:
            plan.append((None,'push',result['projection']))

        plan.append((None,'step','{}_pages'.format(parse_tree[1]),'{}_start'.format(parse_tree[1])))

        if staged:
            plan.append( ('__stage_pages', 'list-pages', ':stage:') )
            plan.append( ('__stage_start', 'label') )
            for col in result['available']:
                plan.append( (col, 'load-table-data', ':stage:', col, '{__stage_pages}') )
            if '__limit__' in scope:
                plan.append((None,'push-limited', result['projection'], scope['__limit__']))
            else:
                plan.append((None,'push',result['projection']))
            plan.append((None,'step','__stage_pages','__stage_start'))
        return result

    elif parse_tree[0] == 'where':
        return build_query_plan(plan, tf, parse_tree[2], scope=dict(scope, __where__=parse_tree[1]))

    elif parse_tree[0] == 'limit':
        result = build_query_plan(plan, tf, parse_tree[2], scope=dict(scope, __postprocess__='limit', __limit__=parse_tree[1]))
        # TODO if we haven't pushed yet.. then push now
        #plan.append(('res','limit-staged-pages',parse_tree[1]))
        #plan.append((None,'push','res'))
        return result

    elif parse_tree[0] == 'group':
        agg_funcs = ['min','max','count','sum','first']
        agg_methods = {}
        def remove_aggregations(pt):
            if pt[0] == 'select':
                projection = []
                for col in pt[1]:
                    if isinstance(col,tuple) and col[0] in agg_funcs:
                        agg_methods[labelize(col[1][0])] = col[0]
                        projection.append(col[1][0])
                    else:
                        projection.append(col)
                return (pt[0], projection)
            else:
                return (pt[0], pt[1], remove_aggregations(pt[2]))
        return build_query_plan(plan, tf, remove_aggregations(parse_tree[2]), scope=dict(scope,__group__=parse_tree[1], __agg__=agg_methods))

    elif parse_tree[0] == 'order':
        result = build_query_plan(plan, tf, parse_tree[2], scope=dict(scope, __sort__=parse_tree[1], __order__=[col[0] if isinstance(col,tuple) and col[1].startswith('!') else col for col in parse_tree[1]], __postprocess__='order'))
        return result
# parse_tree ('limit', 10, ('order', ['ticker'], ('from', 'model_2pm', ('select', ['*']))))

def replace_parse_tree_kwargs(query_tuple, kwargs):
    if isinstance(query_tuple, tuple):
        if query_tuple[0] == ':':
            return kwargs[ query_tuple[1] ]
        else:
            return tuple(replace_parse_tree_kwargs(val, kwargs) for val in query_tuple)
    else:
        return query_tuple

def execute(cb, tf, query_tuple, kwargs):
    query_tuple = replace_parse_tree_kwargs(query_tuple, kwargs)
    plan = []
    build_query_plan(plan, tf, query_tuple)
    import pprint
    pprint.pprint(plan)

    tmp_path = '.staging.glass'
    staging_tf = TableFile(Storage(open(tmp_path, 'wb+')))
    os.unlink(tmp_path)

    execute_plan(cb, tf, staging_tf, plan)
    #table = translate_parse_tree(tf, query_tuple)
    #return execute_query_plan(tf, table)

# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', ['test', '*'])))))
# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', [('.', 'test', '*')])))))
# ('where', ('and', ('like', 'id', (':',)), ('>=', 'date', 20201104)), ('from', 'symbols', ('select', [('count', '*')])))
# ('where', ('like', 'id', (':',)), ('from', 'symbols', ('select', [('count', '*')])))

# defining the execution engine's opcodes:

# select count(*) from z ==>> select count(*) from z group by 1
#   page <- list z
#   start <- label
#   col0 <- load z/$page/{col0}/vals
#   count_any <- reduce $count_any count $a_w by 1
#   step $page $start
#   res <- page {'count(*)': $count_any, 'sum(b)': $sum_b}
#   push $res

# TODO multi-page sorting
# select a, b from z order by c
# select a, b from z where c > 0.3 and c < 0.4 order by c
# select a, b from z where c > 0.3 and c < 0.4 order by d
# select a, sum(b) from z where c > 0.3 and c < 0.4 order by d group by a

# ('group', ['ticker'], ('where', ('and', ('<', 'stats.sd(dow)', 0.005), ('and', ('<', 'stats.sd(woq)', 0.003), ('<', ('.', 'stats', 'theta'), 1.5))), ('from', 'model_2pm', ('select', ['ticker', ('count', '*')]))))

