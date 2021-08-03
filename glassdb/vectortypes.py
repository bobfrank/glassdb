import numpy as np
import datetime
import struct

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


