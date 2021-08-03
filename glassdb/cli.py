from .storage import Storage, type_order, type_names, Directory
from .imports import import_jsonl, import_csv, import_sqlite, import_parquet
from .table import TableFile
from .sql import _sql, sql_completion
from .engine import execute
import traceback
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

    def __init__(self, tf):
        self.tf = tf
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

    def completedefault(self, text, line, begidx, endidx):
        # TODO autocomplete on sql statements would be awesome but a bunch of work.. need to read the parsetab to figure out what state I'm up to.. see if its a table thats next then show table names or column names or sql syntax options...
        #print('"%s"'%line)
        try:
            if line.startswith('.ls '):
                #print('line', line[4:])
                path = line[4:]
                parts = path.split('/')
                start = '/'.join(parts[:-1])
                #print(start, parts[-1])
                vals = self.tf.storage.read_path(start.encode('utf-8'))
                if isinstance(vals, Directory):
                    #print(vals)
                    return [name.decode('utf-8')+('/' if typ == type_order.index('dir') else '') for name, typ in zip(vals.names, vals.types) if name.decode('utf-8').startswith(parts[-1])]
            elif not line.startswith('.'):
                completion = sql_completion(line)
                result = []
                for token, (is_next, is_table, values) in completion.items():
                    if is_table:
                        result = [k.decode('utf-8') for k in self.tf.list_tables()]
                        break
                    else:
                        result.extend(values)
                return result
        except:
            traceback.print_exc()
        #return []

    completenames = completedefault

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
                for name in self.tf.list_tables():
                    print(name.decode('utf-8'))
            elif cmd.lower() == '.memory':
                import psutil
                process = psutil.Process(os.getpid())
                print(int(process.memory_info().rss/1024./1024.),'MB')
            elif cmd.lower().startswith('.ls'):
                def ls(path):
                    vals = self.tf.storage.read_path(path[:-1] if path.endswith(b'/') else path)
                    if isinstance(vals, Directory):
                        for name, typ in zip(vals.names, vals.types):
                            type_name = type_names[typ]
                            print(f"{name.decode('utf-8'):25s} {type_name}")
                    elif vals is not None:
                        print(vals, len(vals))
                    else:
                        print(f'path not found: {path}')
                if cmd.find(' ') < 0:
                    ls(b'')
                else:
                    ls(cmd.split(' ')[1].encode('utf-8'))
            elif cmd.lower().startswith('.schema'):
                def print_schema(table_name):
                    schema = []
                    for col_name, type_name in self.tf.list_columns_and_types(table_name):
                        schema.append('{} {}'.format(col_name.decode('utf-8'), type_name))
                    print('CREATE TABLE {} ({});'.format(table_name.decode('utf-8'), ', '.join(schema)))
                if cmd.find(' ') < 0:
                    for table_name in self.tf.list_tables():
                        print_schema(table_name)
                else:
                    table_name = cmd.lower().split(' ')[1].encode('utf-8')
                    print_schema(table_name)
            else:
                print('unhandled special command', cmd)
        else:
            #print('cmd',cmd)
            s = time.time()
            cache = {}
            try:
                query = _sql(cmd) #"SELECT strategy.name, date, last_trade.SPY, returns.SPY, returns.XLE, returns.XLK FROM strat_returns WHERE strategy.name='up' AND returns.SPY>:minret ORDER BY date")
                def cb(page):
                    s0 = time.time()
                    cache['header'] = self.pprint_table(page, cache.get('header'))
                    cache.setdefault('print_time',0)
                    cache['print_time'] += time.time()-s0
                execute(cb, self.tf, query, {})
            except:
                traceback.print_exc()
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
            storage = Storage(fp_out)
            storage.init()
            tf = TableFile(storage)
            with open(source_path) as fp_in:
                import_csv(tf, fp_in, table_name)

    elif source_path.endswith('.db'):
        with open(glass_path,'wb+') as fp:
            storage = Storage(fp_out)
            storage.init()
            tf = TableFile(storage)
            import_sqlite(tf, source_path)

    elif source_path.endswith('.pq'):
        with open(glass_path,'wb+') as fp:
            storage = Storage(fp)
            storage.init()
            tf = TableFile(storage)
            import_parquet(tf, source_path, table_name)

    elif source_path.endswith('.jsonl'):
        with open(glass_path,'wb+') as fp_out:
            storage = Storage(fp_out)
            storage.init()
            tf = TableFile(storage)
            with open(source_path) as fp_in:
                import_jsonl(tf, fp_in, table_name)

    with open(glass_path,'rb+') as glass_fp:
        tf = TableFile(Storage(glass_fp))
        GlassShell(tf).cmdloop()

if __name__ == '__main__':
    main()

