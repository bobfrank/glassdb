from .storage import init_glassdb, read_path
from .imports import import_jsonl, import_csv, import_sqlite, import_parquet
from .table import table_list_columns_and_types, table_list_tables
from .sql import _sql
from .engine import execute
import time
import cmd
import sys
import os
import os.path
try:
    import readline
except ImportError:
    readline = None

histfile = os.path.expanduser('~/.glassdb_history')
histfile_size = 1000

class GlassShell(cmd.Cmd):
    intro = 'Welcome to the Glass DB SQL shell.\n'
    prompt = 'glassdb> '

    def __init__(self, fp):
        self.fp = fp
        super(GlassShell,self).__init__()

    def preloop(self):
        if readline and os.path.exists(histfile):
            readline.read_history_file(histfile)

    def postloop(self):
        if readline:
            readline.set_history_length(histfile_size)
            readline.write_history_file(histfile)

    def pprint_table(self, page, header=None):
        if header is None:
            header = []
            for col in page:
                header.append(col)
            print('|'.join([x.decode('utf-8') if isinstance(x,bytes) else x for x in header]))
        for row_i in range(len(page[header[0]])):
            print('|'.join([page[col][row_i].decode('utf-8') if isinstance(page[col][row_i],bytes) else str(page[col][row_i]) for col in header]))
        return header

#Cmd.completedefault(text, line, begidx, endidx)¶
#    def complete_add(self, text, line, begidx, endidx):
#        mline = line.partition(' ')[2]
#        offs = len(mline) - len(text)
#        return [s[offs:] for s in completions if s.startswith(mline)]
    def completedefault(self, text, line, begidx, endidx):
        #print('"%s"'%line)
        try:
            if line.startswith('.ls '):
                #print('line', line[4:])
                path = line[4:]
                parts = path.split('/')
                start = '/'.join(parts[:-1])
                #print(start, parts[-1])
                vals = read_path(self.fp, start.encode('utf-8'))
                if isinstance(vals, dict):
                    #print(vals)
                    return [name.decode('utf-8') for name in vals['names'] if name.decode('utf-8').startswith(parts[-1])]
        except:
            import traceback
            traceback.print_exc()
        #return []

    def onecmd(self, cmd):
        if cmd == 'EOF':
            self.postloop()
            os._exit(0)
        cmd = cmd.strip()
        if cmd.endswith(';'):
            cmd = cmd[:-1]
        if cmd == '':
            return
        if cmd.startswith('.'):
            if cmd.lower() == '.tables':
                for name in table_list_tables(fp):
                    print(name.decode('utf-8'))
            elif cmd.lower().startswith('.ls'):
                def ls(path):
                    vals = read_path(self.fp, path)
                    if isinstance(vals,dict) and 'names' in vals:
                        print(vals['names'])
                    else:
                        print(vals, len(vals))
                if cmd.find(' ') < 0:
                    ls(b'')
                else:
                    ls(cmd.split(' ')[1].encode('utf-8'))
            elif cmd.lower().startswith('.schema'):
                def print_schema(table_name):
                    schema = []
                    for col_name, type_name in table_list_columns_and_types(self.fp, table_name):
                        schema.append('{} {}'.format(col_name.decode('utf-8'), type_name))
                    print('CREATE TABLE {} ({});'.format(table_name.decode('utf-8'), ', '.join(schema)))
                if cmd.find(' ') < 0:
                    for table_name in table_list_tables(self.fp):
                        print_schema(table_name)
                else:
                    table_name = cmd.lower().split(' ')[1].encode('utf-8')
                    print_schema(table_name)
            else:
                print('unhandled special command', cmd)
        else:
            #print('cmd',cmd)
            s = time.time()
            query = _sql(cmd) #"SELECT strategy.name, date, last_trade.SPY, returns.SPY, returns.XLE, returns.XLK FROM strat_returns WHERE strategy.name='up' AND returns.SPY>:minret ORDER BY date")
            cache = {}
            def cb(page):
                s0 = time.time()
                cache['header'] = self.pprint_table(page, cache.get('header'))
                cache.setdefault('print_time',0)
                cache['print_time'] += time.time()-s0
            execute(cb, self.fp, query, {})
            timing = time.time() - s - cache.get('print_time',0)
            print('time: {:0.2f}ms'.format(timing*1000))

def main():
    if len(sys.argv) != 2:
        print('Usage: {} [path.{{glass,csv,jsonl,db}}]'.format(sys.argv[0]), file=sys.stderr)
        os._exit(1)

    source_path = sys.argv[1]
    glass_path = '{}.glass'.format(source_path)
    table_name = os.path.basename(source_path).split('.')[0].replace('-','_')
    if source_path.endswith('.glass'):
        glass_path = source_path

    elif source_path.endswith('.csv'):
        with open(glass_path,'wb+') as fp_out:
            init_glassdb(fp_out)
            with open(source_path) as fp_in:
                import_csv(fp_out, fp_in, table_name)

    elif source_path.endswith('.db'):
        with open(glass_path,'wb+') as fp:
            init_glassdb(fp)
            import_sqlite(fp, source_path)

    elif source_path.endswith('.pq'):
        with open(glass_path,'wb+') as fp:
            init_glassdb(fp)
            import_parquet(fp, source_path, table_name)

    elif source_path.endswith('.jsonl'):
        with open(glass_path,'wb+') as fp_out:
            init_glassdb(fp_out)
            with open(source_path) as fp_in:
                import_jsonl(fp_out, fp_in, table_name)

    with open(glass_path,'rb+') as glass_fp:
        GlassShell(glass_fp).cmdloop()

if __name__ == '__main__':
    main()

