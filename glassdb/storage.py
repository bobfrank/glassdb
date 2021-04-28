import lz4.frame
import fcntl
import numpy as np
import time
import struct
import io

type_order = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, 'text', 'dir','file','meta']
type_names = ['u8','u16','u32','u64','i8','i16','i32','i64','f32','f64','text']

class TextVec(object):

    dtype = 'text'

    def __init__(self):
        self.blob = b''
        self.starts = np.array([], dtype=np.uint32)
        self.ends = np.array([], dtype=np.uint32)
        self.text_to_start = {}

    def tobytes(self):
        return struct.pack('!I', len(self.starts)) + self.starts.tobytes() + self.ends.tobytes() + self.blob

    @staticmethod
    def frombuffer(buf):
        nstrs = struct.unpack('!I', buf[:4])[0]
        result = TextVec()
        # saving starts and ends just for filtering to subsets of rows quicker (e.g. by reusing blob and not calculating strlen ever)
        #print(buf[:100])
        #print(len(buf), nstrs, nstrs*4)
        result.starts = np.frombuffer(buf[4:4+4*nstrs], dtype=np.uint32)
        result.ends = np.frombuffer(buf[4+4*nstrs:4+8*nstrs], dtype=np.uint32)
        result.blob = buf[4+8*nstrs:]
        result.text_to_start = None
        return result

    def tolist(self):
        return [self.blob[self.starts[i]:self.ends[i]] for i in range(len(self.starts))]

    def setup_text_to_start(self):
        self.text_to_start = {}
        start_ends = set(zip(self.starts,self.ends))
        for start, end in start_ends:
            self.text_to_start[ self.blob[start:end] ] = start

    def __len__(self):
        return len(self.starts)

    def __repr__(self):
        if len(self) < 15:
            return str(self.tolist())
        else:
            return str([self[i] for i in range(5)] + ['...'] + list(reversed([self[-i] for i in range(5)]))).replace(" '...'", ' ...')

    def __eq__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        idx = self.text_to_start.get(rhs)
        if idx is None:
            return False
        else:
            return self.starts == idx

    def __neq__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        idx = self.text_to_start.get(rhs)
        if idx is None:
            return False
        else:
            return self.starts != idx

# TODO
#    def orderby_order(self):
#        return np.array(sorted(range(len(y)),key=y.__getitem__))

    def groupby_order(self):
        return np.argsort(self.starts)

    def groupby_starts(self):
        prev = np.roll(self.starts,1)
        prev[0] = -1
        return self.starts != prev

    @staticmethod
    def fromlist(ls):
        blob_items = []
        offset = 0
        unique_texts = list(set(ls))
        text_to_start = dict((a,b) for (b,a) in enumerate(unique_texts))
        indexes = np.array([text_to_start[t] for t in ls])
        ends = np.array(np.cumsum(np.array(list(map(len,unique_texts)), dtype=np.uint32)), dtype=np.uint32)
        starts = np.array(np.roll(ends, 1), dtype=np.uint32)
        starts[0] = 0
        blob = b''.join(unique_texts)
        result = TextVec()
        result.blob = blob
        result.starts = starts[indexes]
        result.ends = ends[indexes]
        result.text_to_start = text_to_start
        return result

    def __getitem__(self, index):
        if isinstance(index, int) and not isinstance(index,bool):
            return self.blob[self.starts[index]:self.ends[index]]
        else:
            if isinstance(index,bool) and index == True:
                return self
            result = TextVec()
            result.starts = self.starts[index]
            result.ends = self.ends[index]
            result.blob = self.blob
            result.text_to_start = self.text_to_start
            return result


dir_max_entries = 32
dir_max_name_len = 64

def directory_size():
    # 10KiB
    return dir_max_name_len*dir_max_entries + 8*dir_max_entries + 2*4*dir_max_entries

def serialize_directory(directory):
    names = directory['names']
    offsets = directory['offsets']
    sizes = directory['sizes']
    types = directory['types']
    if len(names) > dir_max_entries:
        raise RuntimeError('too many entries in directory')
    if len(names) < dir_max_entries:
        names.extend([b'']*(dir_max_entries-len(names)))
    if len(offsets) < dir_max_entries:
        offsets = np.concatenate((offsets, np.zeros(dir_max_entries-len(offsets),dtype=np.uint64)))
    if len(sizes) < dir_max_entries:
        sizes = np.concatenate((sizes, np.zeros(dir_max_entries-len(sizes),dtype=np.uint32)))
    if len(types) < dir_max_entries:
        types = np.concatenate((types, np.zeros(dir_max_entries-len(types),dtype=np.uint32)))

    def pad(text):
        if len(text) > dir_max_name_len:
            raise RuntimeError('name is too long')
        return text + b'\0'*(dir_max_name_len-len(text))

    names_data = b''.join([pad(name) for name in names])
    offsets_data = offsets.tobytes()
    sizes_data = sizes.tobytes()
    types_data = types.tobytes()

    return names_data + offsets_data + sizes_data + types_data

def deserialize_directory(buf):
    names = [buf[i*dir_max_name_len:(i+1)*dir_max_name_len].rstrip(b'\0') for i in range(dir_max_entries)]
    off = dir_max_entries*dir_max_name_len
    offsets_size = 8*dir_max_entries
    offsets = np.frombuffer(buf[off:off+offsets_size], dtype=np.uint64)
    off += offsets_size
    sizes_size = 4*dir_max_entries
    sizes = np.frombuffer(buf[off:off+sizes_size], dtype=np.uint32)
    off += sizes_size
    types_size = 4*dir_max_entries
    types = np.frombuffer(buf[off:off+types_size], dtype=np.uint32)
    if b'' in names:
        count = names.index(b'')
        names = names[:count]
        offsets = offsets[:count]
        sizes = sizes[:count]
        types = types[:count]
    return {'names': names, 'offsets': offsets, 'sizes': sizes, 'types': types}

def serialize_vector(data):
    raw_data = data.tobytes()
    res = lz4.frame.compress(raw_data)
    return raw_data if len(res) >= 0.70*len(raw_data) else res

def deserialize_vector(data, type_index):
    if type_order[type_index] == 'text':
        try:
            raw_data = lz4.frame.decompress(data)
        except:
            raw_data = data
        return TextVec.frombuffer(raw_data)
    elif type_order[type_index] in ['file','dir']:
        raise RuntimeError('vector deserialization cannot deserialize the entire file nor directories')
    else:
        s = time.time()
        try:
            raw_data = lz4.frame.decompress(data)
        except:
            raw_data = data
        return np.frombuffer(raw_data, dtype=type_order[type_index])

#def flat_cracking():
#    # not really cracking a full index, once you get to the 5-10k row range its about the same time vs a proper index
#    # idea - partition column into ~100 buckets based on estimated stats, when we add the bucket, we add to the first location available in the total index
#    # storage layout: [left0, left1, ...], [next_free or -1 when full], [left0_index_index, ...], [right0_index_index, ...] (or -1s for unfilled ones), then indexes int column values (in order of which are partitioned first)
#    pass

def init_glassdb(fp):
    free_offsets_data = serialize_vector(np.array([], dtype=np.uint64))
    free_sizes_data = serialize_vector(np.array([], dtype=np.uint64))
    txn_id = 0
    rec = {
        'names': [b'glassdb',b'.free_offsets',b'.free_sizes',b'.txn'],
        'offsets': np.array([0,directory_size(),directory_size()+len(free_offsets_data),txn_id],dtype=np.uint64),
        'sizes': np.array([0, len(free_offsets_data), len(free_sizes_data), 0],dtype=np.uint32),
        'types': np.array([type_order.index('file'), type_order.index(np.uint64), type_order.index(np.uint64), type_order.index('meta')],dtype=np.uint32),
    }
    s = time.time()
    res = serialize_directory(rec)

    fp.seek(0)
    fp.write(res + free_offsets_data + free_sizes_data)

def malloc(fp, size):
    fp.seek(0, io.SEEK_END)
    return fp.tell()

def add_dir_entry(fp, parent_directory_offset, name, offset, size, type_index):
    fp.seek(parent_directory_offset)
    parent_rec = deserialize_directory(fp.read(directory_size()))
    if len(parent_rec['names']) == dir_max_entries:
        if parent_rec['names'][-1] == b'.':
            next_parent_directory_offset = parent_rec['offsets'][-1]
            res = add_dir_entry(fp, next_parent_directory_offset, name, offset, size, type_index)
            if res is not None:
                return res
            else:
                return next_parent_directory_offset

    elif len(parent_rec['names']) >= dir_max_entries-1:
        new_parent_rec = empty_directory()
        new_parent_directory_offset = malloc(fp, directory_size())
        fp.seek(new_parent_directory_offset)
        fp.write(serialize_directory(new_parent_rec))

        parent_rec['names'].append(b'.')
        parent_rec['offsets'] = np.append(parent_rec['offsets'], np.array([new_parent_directory_offset], dtype=np.uint64))
        parent_rec['sizes'] = np.append(parent_rec['sizes'], np.array([directory_size()], dtype=np.uint32))
        parent_rec['types'] = np.append(parent_rec['types'], np.array([type_order.index('dir')], dtype=np.uint32))
        #print('adding',parent_rec['names'], parent_directory_offset)
        fp.seek(parent_directory_offset)
        fp.write(serialize_directory(parent_rec))
        add_dir_entry(fp, new_parent_directory_offset, name, offset, size, type_index)
        return new_parent_directory_offset

    else:
        parent_rec['names'].append(name)
        parent_rec['offsets'] = np.append(parent_rec['offsets'], np.array([offset], dtype=np.uint64))
        parent_rec['sizes'] = np.append(parent_rec['sizes'], np.array([size], dtype=np.uint32))
        parent_rec['types'] = np.append(parent_rec['types'], np.array([type_index], dtype=np.uint32))
        fp.seek(parent_directory_offset)
        fp.write(serialize_directory(parent_rec))
        return parent_directory_offset

def empty_directory():
    return {
        'names': [],
        'offsets': np.array([],dtype=np.uint64),
        'sizes': np.array([],dtype=np.uint32),
        'types': np.array([],dtype=np.uint32),
    }


def mkdir(fp, parent_directory_offset, name, rec=None):
    if rec is None:
        rec = empty_directory()
    new_offset = malloc(fp, directory_size())
    fp.seek(new_offset)
    fp.write(serialize_directory(rec))
    res = add_dir_entry(fp, parent_directory_offset, name, new_offset, directory_size(), type_order.index('dir'))
    return new_offset, res

def mkdir_p(fp, path, rec=None):
    offset = 0
    fp.seek(offset)
    dir_rec = deserialize_directory(fp.read(directory_size()))
    parts = path.split(b'/')
    for i, part in enumerate(parts):
        if part not in dir_rec['names']:
            for part_hat in parts[i:]:
                offset, _ = mkdir(fp, offset, part_hat)
            return offset
        rec_i = dir_rec['names'].index(part)
        offset = dir_rec['offsets'][rec_i]
        fp.seek(offset)
        rec_data = fp.read(dir_rec['sizes'][rec_i])
        if dir_rec['types'][rec_i] == type_order.index('dir'):
            dir_rec = deserialize_directory(rec_data)
        else:
            raise RuntimeError('{} is not a directory'.format(part))
    return offset

def insert_vector(fp, parent_directory_offset, name, values):
    data = serialize_vector(values)
    new_offset = malloc(fp, len(data))
    fp.seek(new_offset)
    fp.write(data)
    add_dir_entry(fp, parent_directory_offset, name, new_offset, len(data), type_order.index(values.dtype))
    return new_offset

cached_dir_records = {}

def read_directory_all(fp, file_offset):
    fp.seek(file_offset)
    dir_rec = deserialize_directory(fp.read(directory_size()))
    if dir_rec['names'] and dir_rec['names'][-1] == b'.':
        next_dir_rec = read_directory_all(fp, dir_rec['offsets'][-1])
        dir_rec['names'] = dir_rec['names'][:-1] + next_dir_rec['names']
        dir_rec['offsets'] = np.concatenate((dir_rec['offsets'][:-1], next_dir_rec['offsets']))
        dir_rec['sizes'] = np.concatenate((dir_rec['sizes'][:-1], next_dir_rec['sizes']))
        dir_rec['types'] = np.concatenate((dir_rec['types'][:-1], next_dir_rec['types']))
    return dir_rec

def read_path(fp, path):
    # TODO so long as we have it cached we must mark it as read-locked
    if () in cached_dir_records:
        dir_rec = cached_dir_records[()]
    else:
        dir_rec = read_directory_all(fp, 0)
        cached_dir_records[()] = dir_rec
    if path == b'':
        return dir_rec
    parts = path.split(b'/')
    for i, part in enumerate(parts):
        if tuple(parts[:i+1]) in cached_dir_records:
            dir_rec = cached_dir_records[tuple(parts[:i+1])]
        else:
            rec_i = dir_rec['names'].index(part)
            offset = dir_rec['offsets'][rec_i]
            if dir_rec['types'][rec_i] == type_order.index('dir'):
                dir_rec = read_directory_all(fp, offset)
                #print('looking at',b'/'.join(parts[:i+1]), dir_rec['names'])
                cached_dir_records[tuple(parts[:i+1])] = dir_rec
            else:
                fp.seek(offset)
                rec_data = fp.read(dir_rec['sizes'][rec_i])
                if i != len(parts)-1:
                    raise RuntimeError('{} is not a directory'.format(part))
                return deserialize_vector(rec_data, dir_rec['types'][rec_i])
    return dir_rec

def open_dir(fp, parent_directory_offset, name):
    dir_rec = read_directory_all(fp, parent_directory_offset)
    if name not in dir_rec['names'] and b'.' in dir_rec['names']:
        return open_dir(fp, dir_rec['offsets'][dir_rec['names'].index(b'.')], name)
    else:
        rec_i = dir_rec['names'].index(name)
        if dir_rec['types'][rec_i] == type_order.index('dir'):
            return dir_rec['offsets'][rec_i]
        else:
            raise RuntimeError('{} is not a directory'.format(name))

