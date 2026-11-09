"""Compressed row mask carried alongside the actual FUA residual packet."""
import struct,zlib
import numpy as np
HEADER=struct.Struct('<4sIII')

def encode_color_point_mask(mask,frame):
    mask=np.asarray(mask)
    if mask.ndim!=1 or mask.dtype!=np.bool_ or isinstance(frame,bool) or not isinstance(frame,int) or frame<1:
        raise ValueError('Invalid point mask')
    rows=np.flatnonzero(mask).astype('<u4')
    if not len(rows):return None
    return HEADER.pack(b'CPM1',frame,len(mask),len(rows))+zlib.compress(rows.tobytes(),9)

def decode_color_point_mask(packet,frame,point_count):
    if len(packet)<HEADER.size:raise ValueError('Truncated point mask')
    magic,got_frame,n,count=HEADER.unpack(packet[:HEADER.size])
    if magic!=b'CPM1' or got_frame!=frame or n!=point_count or count<1 or count>n:
        raise ValueError('Point mask header mismatch')
    obj=zlib.decompressobj()
    try:
        data=obj.decompress(packet[HEADER.size:],count*4+1)
        if obj.unconsumed_tail: raise ValueError('Oversized point mask')
        data+=obj.flush()
    except zlib.error as exc:raise ValueError('Invalid point mask compression') from exc
    if not obj.eof or obj.unused_data or len(data)!=count*4:raise ValueError('Invalid point mask length')
    rows=np.frombuffer(data,dtype='<u4').astype(np.int64)
    if np.any(rows>=n) or np.any(rows[1:]<=rows[:-1]):raise ValueError('Invalid point mask indices')
    mask=np.zeros(n,dtype=np.bool_);mask[rows]=True
    return mask
