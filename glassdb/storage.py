import lz4.frame
import numpy as np
import datetime
import struct
import fcntl
import time
import copy
import io
import os

# TODO transactions (and correct ordering of events)

# text vector and datetime vector

class TextVec(object):
    dtype = 'text'

    def __init__(self):
        self.blob = b''
        self.starts = np.array([], dtype=np.uint32)
        self.ends = np.array([], dtype=np.uint32)
        self.text_to_start = {}
        self.rebuild_blob = False

    def tobytes(self):
        if self.rebuild_blob:
            rep = TextVec.fromlist(self.tolist())
            self.blob = rep.blob
            self.starts = rep.starts
            self.ends = rep.ends
            self.text_to_start = rep.text_to_start
            self.rebuild_blob = rep.rebuild_blob
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
        # pretty slow (400ms+ for 1mm strings with lots of dups)
        s = time.time()
        starts, ends = self.starts, self.ends
        used = set()
        res = {}
        blob = self.blob
        for start, end in zip(starts, ends):
            if start not in used:
                used.add(start)
                res[ blob[start:end] ] = start
        self.text_to_start = res

    def __len__(self):
        return len(self.starts)

    def __repr__(self):
        if len(self) < 15:
            return str(self.tolist())
        else:
            return str([self[i] for i in range(5)] + ['...'] + list(reversed([self[-i-1] for i in range(5)]))).replace(" '...'", ' ...')

    def __eq__(self, rhs):
        # cracked db would be useful here..
        if self.text_to_start is None:
            self.setup_text_to_start()
        idx = self.text_to_start.get(rhs)
        if idx is None:
            return False
        else:
            return self.starts == idx

    def isin(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        idxs = [self.text_to_start.get(r) for r in rhs if self.text_to_start.get(r) is not None]
        if len(idxs) == 0:
            return False
        else:
            return np.isin(self.starts, idxs)

    def __gt__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        accept = np.array([start for text, start in self.text_to_start.items() if text > rhs])
        return np.isin(self.starts, accept)
        #return np.array([self.blob[self.starts[i]:self.ends[i]]>rhs for i in range(len(self.starts))])

    def __ge__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        accept = np.array([start for text, start in self.text_to_start.items() if text >= rhs])
        return np.isin(self.starts, accept)
        #return np.array([self.blob[self.starts[i]:self.ends[i]]>=rhs for i in range(len(self.starts))])

    def __lt__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        accept = np.array([start for text, start in self.text_to_start.items() if text < rhs])
        return np.isin(self.starts, accept)
        #return np.array([self.blob[self.starts[i]:self.ends[i]]<rhs for i in range(len(self.starts))])

    def __le__(self, rhs):
        if self.text_to_start is None:
            self.setup_text_to_start()
        accept = np.array([start for text, start in self.text_to_start.items() if text <= rhs])
        return np.isin(self.starts, accept)
        #return np.array([self.blob[self.starts[i]:self.ends[i]]<=rhs for i in range(len(self.starts))])

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
        text_to_start_idx = dict((a,b) for (b,a) in enumerate(unique_texts))
        idxs = np.array([text_to_start_idx[t] for t in ls])
        ends = np.array(np.cumsum(np.array(list(map(len,unique_texts)), dtype=np.uint32)), dtype=np.uint32)
        starts = np.array(np.roll(ends, 1), dtype=np.uint32)
        starts[0] = 0
        blob = b''.join(unique_texts)
        result = TextVec()
        result.blob = blob
        result.starts = starts[idxs]
        result.ends = ends[idxs]
        result.text_to_start = text_to_start_idx
        return result

    def __getitem__(self, index):
        if isinstance(index, (int,np.int64)) and not isinstance(index,bool):
            return self.blob[self.starts[index]:self.ends[index]]
        else:
            if isinstance(index,bool) and index == True:
                return self
            result = TextVec()
            result.starts = self.starts[index]
            result.ends = self.ends[index]
            result.blob = self.blob
            result.text_to_start = self.text_to_start
            result.rebuild_blob = True
            return result

    @staticmethod
    def concatenate(textvecs):
        result = TextVec()
        offsets = {}
        offset = 0
        for k, textvec in enumerate(textvecs):
            offsets[k] = offset
            offset += len(textvec.blob)
        result.starts = np.concatenate([textvec.starts+offsets[k] for k,textvec in enumerate(textvecs)])
        result.ends = np.concatenate([textvec.ends+offsets[k] for k,textvec in enumerate(textvecs)])
        result.blob = b''.join(textvec.blob for textvec in textvecs)
        result.text_to_start = {}
        return result

class DatetimeVec(object):
    def __init__(self):
        self.datetimes_us = np.array(dtype=np.uint64)

    def tobytes(self):
        return self.datetimes_us.tobytes()

    @staticmethod
    def frombuffer(buf):
        result = DatetimeVec()
        result.datetimes_us = np.frombuffer(buf, dtype=np.uint64)
        return result

    def tolist(self):
        return [dt for dt in self.datetimes_us]

    @staticmethod
    def fromlist(ls):
        result = DatetimeVec()
        result.datetime_us = np.array([int(dt.replace(tzinfo=datetime.timezone.utc).timestamp()*1000000) for dt in ls], dtype=np.uint64)
        return result

    def __getitem__(self, idx):
        return datetime.utcfromtimestamp(self.datetimes_us[idx]/1000000.)

    def __len__(self):
        return len(self.datetimes_us)

    def __repr__(self):
        if len(self) < 15:
            return str(self.tolist())
        else:
            return str([self[i] for i in range(5)] + ['...'] + list(reversed([self[-i-1] for i in range(5)]))).replace(" '...'", ' ...')


type_order = [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, 'text', 'datetime', 'dir', 'file', 'meta']
type_names = ['u8','u16','u32','u64','i8','i16','i32','i64','f32','f64','text','datetime', 'dir', 'file', 'meta']

def serialize_vector(data, disable_compression=False):
    raw_data = data.tobytes()
    if disable_compression:
        return raw_data
    else:
        res = lz4.frame.compress(raw_data)
        return raw_data if len(res) >= 0.70*len(raw_data) else res

def deserialize_vector(data, type_index):
    try:
        raw_data = lz4.frame.decompress(data)
    except:
        raw_data = data
    if type_order[type_index] == 'text':
        return TextVec.frombuffer(raw_data)
    elif type_order[type_index] == 'datetime':
        return DatetimeVec.frombuffer(raw_data)
    elif type_order[type_index] in ['file','dir']:
        raise RuntimeError('vector deserialization cannot deserialize the entire file nor directories')
    else:
        return np.frombuffer(raw_data, dtype=type_order[type_index])


# directory structure

class Directory(object):
    dir_max_entries = 32
    dir_max_name_len = 64

    def __init__(self, names, offsets, sizes, types):
        self.names = names
        self.offsets = offsets
        self.sizes = sizes
        self.types = types

    @staticmethod
    def size():
        return Directory.dir_max_name_len*Directory.dir_max_entries + 8*Directory.dir_max_entries + 2*4*Directory.dir_max_entries

    def serialize(self):
        names = self.names
        offsets = self.offsets
        sizes = self.sizes
        types = self.types
        if len(names) > Directory.dir_max_entries:
            raise RuntimeError('too many entries in directory')
        if len(names) < Directory.dir_max_entries:
            names.extend([b'']*(Directory.dir_max_entries-len(names)))
        if len(offsets) < Directory.dir_max_entries:
            offsets = np.concatenate((offsets, np.zeros(Directory.dir_max_entries-len(offsets),dtype=np.uint64)))
        if len(sizes) < Directory.dir_max_entries:
            sizes = np.concatenate((sizes, np.zeros(Directory.dir_max_entries-len(sizes),dtype=np.uint32)))
        if len(types) < Directory.dir_max_entries:
            types = np.concatenate((types, np.zeros(Directory.dir_max_entries-len(types),dtype=np.uint32)))

        def pad(text):
            if len(text) > Directory.dir_max_name_len:
                raise RuntimeError('name is too long')
            return text + b'\0'*(Directory.dir_max_name_len-len(text))

        names_data = b''.join([pad(name) for name in names])
        offsets_data = offsets.tobytes()
        sizes_data = sizes.tobytes()
        types_data = types.tobytes()

        return names_data + offsets_data + sizes_data + types_data

    @staticmethod
    def frombuffer(buf):
        names = [buf[i*Directory.dir_max_name_len:(i+1)*Directory.dir_max_name_len].rstrip(b'\0') for i in range(Directory.dir_max_entries)]
        off = Directory.dir_max_entries*Directory.dir_max_name_len
        offsets_size = 8*Directory.dir_max_entries
        offsets = np.frombuffer(buf[off:off+offsets_size], dtype=np.uint64)
        off += offsets_size
        sizes_size = 4*Directory.dir_max_entries
        sizes = np.frombuffer(buf[off:off+sizes_size], dtype=np.uint32)
        off += sizes_size
        types_size = 4*Directory.dir_max_entries
        types = np.frombuffer(buf[off:off+types_size], dtype=np.uint32)
        if b'' in names:
            count = names.index(b'')
            names = names[:count]
            offsets = offsets[:count]
            sizes = sizes[:count]
            types = types[:count]
        return Directory(names = names, offsets = offsets, sizes = sizes, types = types)

    @staticmethod
    def empty():
        return Directory(
            names = [],
            offsets = np.array([],dtype=np.uint64),
            sizes = np.array([],dtype=np.uint32),
            types = np.array([],dtype=np.uint32),
        )


# general storage interface

class Storage(object):
    def __init__(self, fp):
        self.fp = fp
        self.cached_vectors = {}
        self.cached_dir_records = {}
        self.cached_dir_offsets = {}


    def init(self):
        free_offsets_data = serialize_vector(np.array([0]*64, dtype=np.uint64))
        free_sizes_data = serialize_vector(np.array([0]*64, dtype=np.uint64))
        txn_id = 0
        rec = Directory(
            names = [b'glassdb',b'.free_offsets',b'.free_sizes',b'.txn'],
            offsets = np.array([0,Directory.size(),Directory.size()+len(free_offsets_data),txn_id],dtype=np.uint64),
            sizes = np.array([0, len(free_offsets_data), len(free_sizes_data), 0],dtype=np.uint32),
            types = np.array([type_order.index('file'), type_order.index(np.uint64), type_order.index(np.uint64), type_order.index('meta')],dtype=np.uint32),
        )
        res = rec.serialize()

        self.fp.seek(0)
        self.fp.write(res + free_offsets_data + free_sizes_data)


    def _malloc(self, size):
        self.fp.seek(0, io.SEEK_END)
        return self.fp.tell()
        # TODO look through the _free offsets and sizes to grab from...


    def _free(self, offset, size):
        pass
        # TODO if new offset/size are at the end then just truncate file

        # load _free offsets/sizes
        # if we would go beyond the existing 0's to do that, then double the size and add the current _free offsets/sizes's vector offset and size to the list as well
        # add new offset/size to the list
        # save _free offsets/sizes (either in place or to a new location if we resized)

        #free_offsets_data = serialize_vector(np.array([0]*64, dtype=np.uint64))
        #free_sizes_data = serialize_vector(np.array([0]*64, dtype=np.uint64))


    def _add_dir_entry(self, parent_dir_offset, name, offset, size, type_index):
        self.fp.seek(parent_dir_offset)
        parent_rec = Directory.frombuffer(self.fp.read(Directory.size()))
        if len(parent_rec.names) == Directory.dir_max_entries:
            if parent_rec.names[-1] == b'.':
                next_parent_directory_offset = parent_rec.offsets[-1]
                res = self._add_dir_entry(next_parent_directory_offset, name, offset, size, type_index)
                if res is not None:
                    return res
                else:
                    return next_parent_directory_offset

        elif len(parent_rec.names) >= Directory.dir_max_entries-1:
            new_parent_rec = Directory.empty()
            new_parent_directory_offset = self._malloc(Directory.size())
            self.fp.seek(new_parent_directory_offset)
            self.fp.write(new_parent_rec.serialize())

            parent_rec.names.append(b'.')
            parent_rec.offsets = np.append(parent_rec.offsets, np.array([new_parent_directory_offset], dtype=np.uint64))
            parent_rec.sizes = np.append(parent_rec.sizes, np.array([Directory.size()], dtype=np.uint32))
            parent_rec.types = np.append(parent_rec.types, np.array([type_order.index('dir')], dtype=np.uint32))
            #print('adding',parent_rec['names'], parent_dir_offset)
            self.fp.seek(parent_dir_offset)
            self.fp.write(parent_rec.serialize())
            self._add_dir_entry(new_parent_directory_offset, name, offset, size, type_index)
            return new_parent_directory_offset

        else:
            parent_rec.names.append(name)
            parent_rec.offsets = np.append(parent_rec.offsets, np.array([offset], dtype=np.uint64))
            parent_rec.sizes = np.append(parent_rec.sizes, np.array([size], dtype=np.uint32))
            parent_rec.types = np.append(parent_rec.types, np.array([type_index], dtype=np.uint32))
            self.fp.seek(parent_dir_offset)
            self.fp.write(parent_rec.serialize())
            return parent_dir_offset


    def _mkdir(self, parent_dir_offset, name, rec=None):
        if rec is None:
            rec = Directory.empty()
        new_offset = self._malloc(Directory.size())
        self.fp.seek(new_offset)
        self.fp.write(rec.serialize())
        res = self._add_dir_entry(parent_dir_offset, name, new_offset, Directory.size(), type_order.index('dir'))
        return new_offset, res


    def insert_vector(self, path, name, values, disable_compression=False):
        # create parent directories if not exists
        self.read_path(path)
        parts = path.split(b'/')
        parent_dir_offset = 0
        for i,part in enumerate(parts):
            part_path = b'/'.join(parts[:i+1])
            if part_path in self.cached_dir_offsets:
                parent_dir_offset = self.cached_dir_offsets[part_path]
            else:
                parent_dir_offset, new_parent_offset = self._mkdir(parent_dir_offset, part)
                self.cached_dir_records[part_path] = Directory.empty()
                self.cached_dir_offsets[part_path] = parent_dir_offset
                parent_path = b'/'.join(parts[:i])
                self.cached_dir_offsets[parent_path] = new_parent_offset # if we needed to create a new directory structure for this path
                parent_rec = self.cached_dir_records[parent_path]
                parent_rec.names.append(name)
                parent_rec.types = np.concatenate((parent_rec.types, np.array([type_order.index('dir')], dtype=np.uint64)))
                parent_rec.offsets = np.concatenate((parent_rec.offsets, np.array([parent_dir_offset], dtype=np.uint32)))
                parent_rec.sizes = np.concatenate((parent_rec.sizes, np.array([Directory.size()], dtype=np.uint32)))

        # save the actual vector
        self.cached_vectors[path + b'/' + name] = values
        data = serialize_vector(values, disable_compression=disable_compression)
        new_offset = self._malloc(len(data))
        self.fp.seek(new_offset)
        self.fp.write(data)
        # note it in the parent
        self._add_dir_entry(parent_dir_offset, name, new_offset, len(data), type_order.index(values.dtype))
        return new_offset


    def update_vector(self, path, name, new_vals, disable_compression=False):
        self.cached_vectors[path + b'/' + name] = new_vals
        data = serialize_vector(new_vals, disable_compression=disable_compression)

        page_rec = self.read_path(path)
        vals_idx = page_rec.names.index(name)
        offset = page_rec.offsets[vals_idx]
        size = page_rec.sizes[vals_idx]
        if len(data) > size:
            self._free(offset, size)
            new_offset = self._malloc(len(data))
            self.fp.seek(new_offset)
            self.fp.write(data)

            if path in self.cached_dir_records:
                # otherwise we need to read it and update it and save it back.. TODO
                dir_rec = self.cached_dir_records[path]
                dir_rec.offsets = copy.copy(dir_rec.offsets)
                dir_rec.sizes = copy.copy(dir_rec.sizes)
                dir_rec.offsets[vals_idx] = new_offset
                dir_rec.sizes[vals_idx] = len(data)
                self.fp.seek(self.cached_dir_offsets[path])
                self.fp.write(dir_rec.serialize())
        else:
            self.fp.seek(offset)
            self.fp.write(data)
            if len(data) < size:
                if path in self.cached_dir_records:
                    # otherwise we need to read it and update it and save it back.. TODO
                    dir_rec = self.cached_dir_records[path]
                    dir_rec.sizes = copy.copy(dir_rec.sizes)
                    dir_rec.sizes[vals_idx] = len(data)
                    self.fp.seek(self.cached_dir_offsets[path])
                    self.fp.write(dir_rec.serialize())
                self._free(offset+len(data), size-len(data))

    def _read_directory_all(self, file_offset):
        self.fp.seek(file_offset)
        dir_rec = Directory.frombuffer(self.fp.read(Directory.size()))
        if len(dir_rec.names) and dir_rec.names[-1] == b'.':
            next_dir_rec = self._read_directory_all(dir_rec.offsets[-1])
            dir_rec.names = dir_rec.names[:-1] + next_dir_rec.names
            dir_rec.offsets = np.concatenate((dir_rec.offsets[:-1], next_dir_rec.offsets))
            dir_rec.sizes = np.concatenate((dir_rec.sizes[:-1], next_dir_rec.sizes))
            dir_rec.types = np.concatenate((dir_rec.types[:-1], next_dir_rec.types))
        return dir_rec

    def read_path(self, path):
        # TODO so long as we have it cached we must mark it as read-locked
        if path in self.cached_vectors:
            return self.cached_vectors[path]
        if b'' in self.cached_dir_records:
            dir_rec = self.cached_dir_records[b'']
        else:
            dir_rec = self._read_directory_all(0)
            self.cached_dir_records[b''] = dir_rec
            self.cached_dir_offsets[b''] = 0
        if path == b'':
            return dir_rec
        parts = path.split(b'/')
        for i, part in enumerate(parts):
            if b'/'.join(parts[:i+1]) in self.cached_dir_records:
                dir_rec = self.cached_dir_records[b'/'.join(parts[:i+1])]
            else:
                if part not in dir_rec.names:
                    return None
                rec_i = dir_rec.names.index(part)
                offset = dir_rec.offsets[rec_i]
                if dir_rec.types[rec_i] == type_order.index('dir'):
                    dir_rec = self._read_directory_all(offset)
                    #print('looking at',b'/'.join(parts[:i+1]), dir_rec['names'])
                    self.cached_dir_records[b'/'.join(parts[:i+1])] = dir_rec
                    self.cached_dir_offsets[b'/'.join(parts[:i+1])] = offset
                elif dir_rec.types[rec_i] == type_order.index('meta'):
                    return np.array([offset])
                else:
                    self.fp.seek(offset)
                    s0 = time.time()
                    rec_data = self.fp.read(dir_rec.sizes[rec_i])
                    if i != len(parts)-1:
                        raise RuntimeError('{} is not a directory'.format(part))
                    res = deserialize_vector(rec_data, dir_rec.types[rec_i])
                    self.cached_vectors[path] = res
                    return res
        return dir_rec


#    def set_lock_offset(self, offset, size, write_lock=False):
#        l_type = fcntl.F_WRLCK if write_lock else fcntl.F_RDLCK
#        if sys.platform.startswith('aix'):
#          # Python on AIX is compiled with LARGEFILE support, which changes the
#          # struct size.
#          op = struct.pack('hhIllqq', l_type, os.SEEK_SET, offset, size, 0, 0, 0)
#        else:
#          op = struct.pack('hhllhhl', l_type, os.SEEK_SET, offset, size, 0, 0, 0)
#        return fcntl.fcntl(fd, fcntl.F_OFD_SETLK, op)
#
#
#    def unset_lock_offset(self, offset, size):
#        if sys.platform.startswith('aix'):
#          # Python on AIX is compiled with LARGEFILE support, which changes the
#          # struct size.
#          op = struct.pack('hhIllqq', fcntl.F_UNLCK, os.SEEK_SET, offset, size, 0, 0, 0)
#        else:
#          op = struct.pack('hhllhhl', fcntl.F_UNLCK, os.SEEK_SET, offset, size, 0, 0, 0)
#        return fcntl.fcntl(fd, fcntl.F_OFD_SETLK, op)
#
#    def set_lock(self, path, write_lock=False):
#        pass
#
#    def unset_lock(self, path):
#        pass
#
#    def rename_vector(self, path, new_name):
#        pass
#
#    def unlink_vector(self, path):
#        pass
#
#
#class TxnStorage(Storage):
#    def __init__(self, fp):
#        super(TxnStorage,self).__init__(fp)
#        self.txn = None
#        self.read_locked = None
#
#
#    def begin_transaction(self):
#        if self.txn is not None:
#            raise RuntimeError()
#
#    def commit_transaction(self):
#        if self.txn is None:
#            raise RuntimeError('must begin a transaction before commiting it')
#
#    def abort_transaction(self):
#        if self.txn is None:
#            raise RuntimeError('must begin a transaction before aborting it')
#
#
#    def insert_vector(self, path, name, values, disable_compression=False):
#        if self.txn is None:
#            raise RuntimeError('must begin a transaction before inserting a vector')
#        super(TxnStorage,self).insert_vector(path, name, values, disable_compression=disable_compression)
#
#    def update_vector(self, path, name, new_vals, disable_compression=False):
#        if self.txn is None:
#            raise RuntimeError('must begin a transaction before updating a vector')
#        super(TxnStorage,self).update_vector(path, name, new_vals, disable_compression=disable_compression)
#
#
#    def lock_read_latest(self):
#        pass
#
#    def unlock_read(self):
#        pass
#
#    def read_path(self, path):
#        if self.read_locked is None:
#            raise RuntimeError('must lock_read_latest() before calling read_path()')
#        super(TxnStorage,self).read_path(path)
#
