from storage import init_glassdb, read_path
from imports import import_jsonl, import_csv, import_sqlite
from sql import _sql
from engine import execute
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

    def pprint_table(self, res):
        header = None
        for page in res:
            if header is None:
                header = []
                for col in res[page]:
                    header.append(col)
                print('|'.join([x.decode('utf-8') for x in header]))
            for row_i in range(len(res[page][header[0]])):
                print('|'.join([res[page][col][row_i].decode('utf-8') if isinstance(res[page][col][row_i],bytes) else str(res[page][col][row_i]) for col in header]))

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
                for name in read_path(self.fp, b'')['names'][1:]:
                    if name.startswith(b'.'):
                        continue
                    print(name.decode('utf-8'))
            else:
                print('unhandled special command', cmd)
        else:
            #print('cmd',cmd)
            query = _sql(cmd) #"SELECT strategy.name, date, last_trade.SPY, returns.SPY, returns.XLE, returns.XLK FROM strat_returns WHERE strategy.name='up' AND returns.SPY>:minret ORDER BY date")
            #print(query)
            res = execute(self.fp, query, {})
            self.pprint_table(res)

def main():
    if len(sys.argv) != 2:
        print('Usage: {} [path.{{glass,csv,jsonl,db}}]'.format(sys.argv[0]), file=sys.stderr)
        os._exit(1)

    source_path = sys.argv[1]
    if source_path.endswith('.glass'):
        glass_path = source_path

    elif source_path.endswith('.csv'):
        glass_path = '{}.glass'.format(source_path)
        with open(glass_path,'wb+') as fp_out:
            init_glassdb(fp_out)
            with open(source_path) as fp_in:
                import_csv(fp_out, fp_in, os.path.basename(source_path).split('.')[0].replace('-','_'))

    elif source_path.endswith('.db'):
        glass_path = '{}.glass'.format(source_path)
        with open(glass_path,'wb+') as fp:
            init_glassdb(fp)
            with sqlite3.connect(source_path) as conn:
                import_sqlite(fp, conn)

    elif source_path.endswith('.jsonl'):
        glass_path = '{}.glass'.format(source_path)
        with open(glass_path,'wb+') as fp_out:
            init_glassdb(fp_out)
            with open(source_path) as fp_in:
                import_jsonl(fp_out, fp_in, os.path.basename(source_path).split('.')[0].replace('-','_'))

    with open(glass_path,'rb+') as glass_fp:
        GlassShell(glass_fp).cmdloop()

if __name__ == '__main__':
    main()

