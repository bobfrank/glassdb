import numpy as np
from .sql import _sql
from .storage import TextVec
from .table import table_list_pages, table_load_data, table_list_columns
import time
import copy
import string

# TODO joins are complicated
# TODO all these encode('utf-8')'s need to be cleaned up (into one place), also make sure unicode all works well inside sql too

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

def execute_plan(cb, fp, plan):
    ip = 0
    values = {}
    s = time.time()
    while ip < len(plan):
        cmd = plan[ip]
        print('cmd',cmd)
        s = time.time()
        result_name = cmd[0]
        method = cmd[1]
        args = []
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
                ip = values[args[1].decode('utf-8')] # going to be incremented before ip is executed.. but points to 'label' command, so good ok skip that op

        elif method == 'label':
            values[result_name] = ip
            print(result_name, ip)

        elif method == 'list_pages':
            table_name = args[0]
            names = table_list_pages(fp, table_name)
            values[result_name] = {'items': names, 'index': 0}
            if len(names) == 0:
                while ip+1 < len(plan) and plan[ip+1][1] != 'step':
                    ip += 1

        elif method == 'load_data':
            #plan.append( (col, 'load_data', scope['__table__'], col, f'{scope["__table__"]}_pages') )
            table_name = args[0]
            col = args[1]
            page = args[2]
            values[result_name] = table_load_data(fp, table_name, col, page)

        elif method == 'condition':
            def replace_values(cond):
                if isinstance(cond, tuple) and cond[0] == '.' and '.'.join(cond[1:]) in values:
                    return values['.'.join(cond[1:])]
                elif isinstance(cond, str) and cond in values:
                    return values[cond]
                elif isinstance(cond, tuple):
                    return tuple(replace_values(x) for x in cond)
                else:
                    return cond
            values[result_name] = evaluate_conditions(replace_values(cmd[2]))

        elif method == 'where':
            values[result_name] = values[cmd[2]][values[cmd[3]]]

        # TODO __staged__ needs to be in a temp file (and able to handle a bunch of pages being sent)
        elif method == 'stage':
            if len(cmd) == 4:
                values['__staged__'] = dict((var,values[var]) for var in cmd[2])
                if len(values['__staged__']) and len(list(values['__staged__'].values())[0]) >= cmd[3]:
                    ip += 1
            else:
                values['__staged__'] = dict((var,values[var]) for var in cmd[2])
    
        elif method == 'group-indexes':
            group_by = cmd[2]
            if len(group_by) != 1:
                raise NotImplementedError('Only can handle group by for one parameter for now')
            for col in group_by:
                if isinstance(values[col], TextVec):
                    projection = values[col].groupby_order()
                    gbo = values[col][projection]
                    start_flag = gbo.groupby_starts()
                    start_indexes = np.concatenate(([0], np.flatnonzero(start_flag)+1))
                    print('start_flag', start_flag)
                    print('start_indexes', start_indexes)
                    values[result_name] = {'project': projection, 'start_indexes': start_indexes}
                else:
                    projection = np.argsort(values[col])
                    gbo = values[col][projection]
                    vhat = np.roll(gbo,1)
                    vhat[0] = -1
                    start_flag = vhat != gbo
                    start_indexes = np.concatenate(([0], np.flatnonzero(start_flag)+1))
                    values[result_name] = {'project': projection, 'start_indexes': start_indexes}

        elif method == 'reduce-to-stage':
            group_by = cmd[2]
            agg_func = cmd[3]
            agg_col = cmd[4]
            final_name = '{}({})'.format(agg_func, agg_col)
            agg_value = values[ agg_col ]
            agg_value_ordered = agg_value[ values[group_by]['project'] ]
            start_indexes = values[group_by]['start_indexes']
            if agg_func == 'first':
                result = agg_value_ordered[start_indexes]
            elif agg_func == 'sum':
                result = np.add.reduceat(agg_value_ordered, start_indexes)
            else:
                raise NotImplementedError('havent implemented fn `{}` yet'.format(agg_func))
            values.setdefault('__staged__',{})[final_name] = result

        elif method == 'sort-staged-pages':
            print(values['__staged__'])
            pass #cmd ('res', 'sort-staged-pages', [('ticker', '!desc')])
#                order = table['order']
#                if len(pages) != 1:
#                    raise RuntimeError('multi-page sorting is not implemented yet')
#                def extract_order_vec(x):
#                    if isinstance(x,tuple) and x[-1] == '!asc':
#                        return extract_order_vec(x[0])
#                    elif isinstance(x,tuple) and x[-1] == '!desc':
#                        return -extract_order_vec(x[0])
#                    else:
#                        return page_vectors['.'.join(x[1:]).encode('utf-8') if isinstance(x,tuple) else x.encode('utf-8')]
#                #s = time.time()
#                idx = np.lexsort( tuple(extract_order_vec(x) for x in reversed(order)) )
#                #print('first',time.time()-s)
#                #s = time.time()
#                #idx_hat = np.lexsort( tuple(extract_order_vec(x)[idx] for x in reversed(order)) )
#                #print('pre-sorted',time.time()-s)
#                #s = time.time()
#                #idx_hat = np.lexsort( tuple(np.hstack((extract_order_vec(x)[idx],extract_order_vec(x)[idx])) for x in reversed(order)) )
#                #print('pre-sorted',time.time()-s)
#                if page in result:
#                    for col in result[page]:
#                        result[page][col] = result[page][col][idx]

        elif method == 'limit-staged-pages':
            for key in values['__staged__']:
                values['__staged__'][key] = values['__staged__'][key][:cmd[2]]
            values[result_name] = values['__staged__']

        elif method == 'push':
            if not isinstance(cmd[2], list) and  cmd[2] in values:
                cb( values[cmd[2]] )
            else:
                cb( dict((var,values[var]) for var in cmd[2]) )

        ip += 1
        print('   ',time.time()-s)

# TODO cracking
# select a,b from z where c >= 0.3 and c <= 0.4
#   page <- list z
#   start <- label
#   a <- load z/$page/a/vals
#   b <- load z/$page/b/vals
#   c <- load z/$page/c/vals
#   c_idx_pct <- list z/$page/c/idx/pct
#   c_paths <- crack-paths $c_idx_pct (__ >= 0.3 & __ <= 0.4)
#   c_idx <- load $c_paths
#   c_idx_updated <- crack-build $c $c_idx {probably need to pass c_paths or something... TODO}
#   save $c_paths $c_idx_updated # only save if the data has changed (so if we sorted previous crack or if we just cracked for the first time)
#   idx <- crack-condition $c $c_idx_updated (__ >= 0.3 & __ <= 0.4)
#   a_w <- where $a $idx
#   b_w <- where $b $idx
#   res <- page {'a': $a_w, 'b': $b_w}
#   push $res
#   step $page $start

def build_query_plan(plan, fp, parse_tree, scope=None):
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
                if isinstance(tree, tuple) and tree[0] in ['and','or']:
                    extract_filter_columns(tree[1])
                    extract_filter_columns(tree[2])
                elif isinstance(tree, tuple) and tree[0] == '.':
                    filter_cols.append('.'.join(tree[1:]))
                else:
                    filter_cols.append(tree[1])
                    filter_cols.append(tree[2])
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
            columns = [col.decode('utf-8') for col in table_list_columns(fp, scope['__table__'].encode('utf-8'))]
            if '*' in projection or '{}.*'.format(scope['__table__']) in projection:
                projection = projection + list(map(labelize,columns))
            cols = set(projection).union(map(labelize,filter_cols)).union(map(labelize,group_cols)).union(map(labelize,order_cols))
            for col in projection:
                if col in columns and col in cols:
                    res.append(col)
            for col in cols:
                if col in columns:
                    plan.append( (col, 'load_data', scope['__table__'], col, '{%s_pages}'%scope['__table__']) )
        return res

    elif parse_tree[0] == 'from':
        table_name = parse_tree[1]
        plan.append( ('{}_pages'.format(table_name), 'list_pages', table_name) )
        plan.append( ('{}_start'.format(table_name), 'label') )
        args = build_query_plan(plan, fp, parse_tree[2], scope=dict(scope,__table__=table_name))

        if '__where__' in scope:
            plan.append(('idx','condition',scope['__where__']))
            for arg in args:
                plan.append((arg, 'where',arg,'idx'))

        if scope is None:
            plan.append((None,'push',args))
        elif '__group__' in scope:
            plan.append(('__group__', 'group-indexes', scope['__group__']))
            for arg in args:
                plan.append((None, 'reduce-to-stage', '__group__', scope['__agg__'].get(arg,'first'), arg))
        elif '__postprocess__' in scope:
            if scope['__postprocess__'] == 'limit':
                plan.append((None,'stage',args, scope['__limit__']))
            else:
                plan.append((None,'stage',args))
        else:
            plan.append((None,'push',args))

        plan.append((None,'step','{}_pages'.format(parse_tree[1]),'{}_start'.format(parse_tree[1])))
        return args

    elif parse_tree[0] == 'where':
        return build_query_plan(plan, fp, parse_tree[2], scope=dict(scope, __where__=parse_tree[1]))

    elif parse_tree[0] == 'limit':
        args = build_query_plan(plan, fp, parse_tree[2], scope=dict(scope, __postprocess__='limit', __limit__=parse_tree[1]))
        plan.append(('res','limit-staged-pages',parse_tree[1]))
        plan.append((None,'push','res'))
        return args

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
        return build_query_plan(plan, fp, remove_aggregations(parse_tree[2]), scope=dict(scope,__group__=parse_tree[1], __agg__=agg_methods))

    elif parse_tree[0] == 'order':
        args = build_query_plan(plan, fp, parse_tree[2], scope=dict(scope, __order__=[col[0] if isinstance(col,tuple) and col[1].startswith('!') else col for col in parse_tree[1]], __postprocess__='order'))
        plan.append((None,'sort-staged-pages',parse_tree[1]))
        return args
# parse_tree ('limit', 10, ('order', ['ticker'], ('from', 'model_2pm', ('select', ['*']))))

def replace_parse_tree_kwargs(query_tuple, kwargs):
    if isinstance(query_tuple, tuple):
        if query_tuple[0] == ':':
            return kwargs[ query_tuple[1] ]
        else:
            return tuple(replace_parse_tree_kwargs(val, kwargs) for val in query_tuple)
    else:
        return query_tuple

def execute(cb, fp, query_tuple, kwargs):
    query_tuple = replace_parse_tree_kwargs(query_tuple, kwargs)
    plan = []
    build_query_plan(plan, fp, query_tuple)
    import pprint
    pprint.pprint(plan)
    execute_plan(cb, fp, plan)
    #table = translate_parse_tree(fp, query_tuple)
    #return execute_query_plan(fp, table)

# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', ['test', '*'])))))
# ('order', ['other'], ('group', ['xyz'], ('where', ('==', 'a', (':', 'a')), ('from', 'magic', ('select', [('.', 'test', '*')])))))
# ('where', ('and', ('like', 'id', (':',)), ('>=', 'date', 20201104)), ('from', 'symbols', ('select', [('count', '*')])))
# ('where', ('like', 'id', (':',)), ('from', 'symbols', ('select', [('count', '*')])))

#if __name__ == '__main__':
#    s = time.time()
#    query = _sql("SELECT strategy.name, date, last_trade.SPY, returns.SPY, returns.XLE, returns.XLK FROM strat_returns WHERE strategy.name='up' AND returns.SPY>:minret ORDER BY date")
#    with open('strat_returns.glass', 'rb') as fp:
#        res = execute(fp, query, {'minret': 0.005})
#    print(time.time()-s)
#    print(res)
#    print(len(res[b'\0\0\0\0pg'][b'date']))

# TODO expose (1) as a library (to be used like sqlite or pandas), (2) as a server (to be used like mssql or clickhouse), and (3) as a command line interface to use sql queries on arbitrary files that can be imported

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

# k-way merge
#while number of files > 1
#    fileList = Load all file names
#    i = 0
#    while i < fileList.length
#        filesToMerge = copy files i through i+k-1 from file list
#        merge(filesToMerge, output file name)
#        i += k
#    end while
#end while

# ('group', ['ticker'], ('where', ('and', ('<', 'stats.sd(dow)', 0.005), ('and', ('<', 'stats.sd(woq)', 0.003), ('<', ('.', 'stats', 'theta'), 1.5))), ('from', 'model_2pm', ('select', ['ticker', ('count', '*')]))))

