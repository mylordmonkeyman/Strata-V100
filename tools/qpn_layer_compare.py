#!/usr/bin/env python3
"""Compare existing private prefill feature dumps; export aggregates only.

STRATA_DBG_FEAT records int32 {layer,position,tokens,has_hidden}, then
uint16 routed IDs, BF16 route weights, and optional hidden FP16 bits on sm70.
No arrays, raw values, paths, or token/expert IDs are returned by this module.
"""
import hashlib
import math
import struct
import numpy as np

N, K, LAYERS = 2560, 10, 48


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise ValueError('truncated private feature dump')
    return b


def records(path):
    seen = set()
    with open(path, 'rb') as f:
        while True:
            header = f.read(16)
            if not header:
                return
            if len(header) != 16:
                raise ValueError('truncated private feature header')
            l, pos, t, hidden = struct.unpack('<4i', header)
            if not (0 <= l < LAYERS and 0 <= pos <= 4096 and 1 <= t <= 1024 and hidden == 1):
                raise ValueError('invalid private feature geometry')
            key = (pos, l)
            if key in seen:
                raise ValueError('duplicate private feature record')
            seen.add(key)
            ids = read_exact(f, t*K*2)
            weights = read_exact(f, t*K*2)
            x = read_exact(f, t*N*2)
            yield key, t, ids, weights, x


def stats(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or mismatched private activation data')
    d = a.astype(np.float64)-b.astype(np.float64)
    norm = float(np.sum(a.astype(np.float64)**2))
    err = float(np.sum(d*d))
    return dict(relative_l2=math.sqrt(err/norm) if norm else (0. if not err else None),
                max_abs=float(np.max(np.abs(d))), bitwise_equal=a.tobytes()==b.tobytes())


def compare_features(a_path, b_path):
    a, b = iter(records(a_path)), iter(records(b_path))
    result = []
    while True:
        ar, br = next(a, None), next(b, None)
        if ar is None or br is None:
            if ar is not None or br is not None:
                raise ValueError('missing private feature records')
            break
        ka, t, ai, aw, ax = ar
        kb, u, bi, bw, bx = br
        if ka != kb or t != u:
            raise ValueError('private feature record order or shape changed')
        xmetrics = stats(np.frombuffer(ax, '<f2'), np.frombuffer(bx, '<f2'))
        # BF16 route-weight conversion is exact via the high bits of float32.
        av = (np.frombuffer(aw, '<u2').astype(np.uint32) << 16).view(np.float32)
        bv = (np.frombuffer(bw, '<u2').astype(np.uint32) << 16).view(np.float32)
        wmetrics = stats(av, bv)
        result.append(dict(chunk=ka[0], layer=ka[1], tokens=t,
                           route_ids_equal=ai==bi,
                           route_mismatch_entries=int(np.count_nonzero(np.frombuffer(ai, '<u2') != np.frombuffer(bi, '<u2'))),
                           hidden=xmetrics, route_weights=wmetrics,
                           hidden_sha256_a=hashlib.sha256(ax).hexdigest(), hidden_sha256_b=hashlib.sha256(bx).hexdigest()))
    expected = {(pos,l) for pos in (0,1024) for l in range(LAYERS)}
    if ({(r['chunk'],r['layer']) for r in result} != expected or
            any(r['tokens'] != (1024 if r['chunk']==0 else 1023) for r in result)):
        raise ValueError('incomplete 2047-token private feature coverage')
    first_hidden = next((dict(chunk=r['chunk'], layer=r['layer']) for r in result if not r['hidden']['bitwise_equal']), None)
    first_route = next((dict(chunk=r['chunk'], layer=r['layer']) for r in result if not r['route_ids_equal']), None)
    return dict(scope='synchronized private feature capture; not timing-transparent', hidden_encoding='IEEE FP16 on pinned sm70',
                records=result, first_hidden_difference=first_hidden, first_route_difference=first_route)


def compare_residuals(a_path, b_path):
    result = []
    width = 4*N  # pinned HC=4
    with open(a_path, 'rb') as a, open(b_path, 'rb') as b:
        for pos, t in ((0,1024),(1024,1023)):
            av = (np.frombuffer(read_exact(a,t*width*2), '<u2').astype(np.uint32)<<16).view(np.float32)
            bv = (np.frombuffer(read_exact(b,t*width*2), '<u2').astype(np.uint32)<<16).view(np.float32)
            result.append(dict(chunk=pos,tokens=t,encoding='BF16 diagnostic snapshot',**stats(av,bv)))
        if a.read(1) or b.read(1):
            raise ValueError('unexpected private residual dump tail')
    return result


# phase -> (layer, columns, encoding, aggregate label)
BOUNDARIES = {
    6:(2,N,1,'layer2_moe_output'), 7:(2,4,1,'layer2_hc_injection'), 8:(2,4*N,1,'layer2_exit_residual'),
    0:(3,4*N,1,'layer3_entry_residual'), 1:(3,4,1,'layer3_entry_norm_scales'),
    2:(3,4*N,2,'layer3_entry_norm_image'), 3:(3,N,1,'layer3_attention_input'),
    4:(3,N,1,'layer3_attention_output'), 9:(3,4,1,'layer3_hc_injection'), 5:(3,4*N,1,'layer3_post_attention_residual')}


def boundary_records(path):
    seen = set()
    with open(path, 'rb') as f:
        while True:
            header = f.read(64)
            if not header:
                return
            if len(header) != 64:
                raise ValueError('truncated private boundary header')
            version,pos,l,phase,t,cols,encoding,count=struct.unpack('<8q',header)
            if (version != 1 or pos not in (0,1024) or not 1 <= t <= 1024 or
                    phase not in BOUNDARIES or (l,cols,encoding) != BOUNDARIES[phase][:3] or count != t*cols):
                raise ValueError('invalid private boundary geometry')
            key=(pos,l,phase)
            if key in seen:
                raise ValueError('duplicate private boundary record')
            seen.add(key)
            yield key,t,encoding,read_exact(f,count*(4 if encoding==1 else 2))


def compare_boundaries(a_path,b_path):
    a,b=iter(boundary_records(a_path)),iter(boundary_records(b_path))
    rows=[]
    while True:
        ar,br=next(a,None),next(b,None)
        if ar is None or br is None:
            if ar is not None or br is not None:
                raise ValueError('missing private boundary record')
            break
        key,t,encoding,ax=ar
        kb,u,eb,bx=br
        if key != kb or t != u or encoding != eb:
            raise ValueError('private boundary order or shape changed')
        dtype='<f4' if encoding==1 else '<f2'
        rows.append(dict(chunk=key[0],layer=key[1],phase=key[2],label=BOUNDARIES[key[2]][3],tokens=t,
                         sha256_a=hashlib.sha256(ax).hexdigest(),sha256_b=hashlib.sha256(bx).hexdigest(),
                         **stats(np.frombuffer(ax,dtype),np.frombuffer(bx,dtype))))
    expected={(pos,meta[0],phase) for pos in (0,1024) for phase,meta in BOUNDARIES.items()}
    if ({(r['chunk'],r['layer'],r['phase']) for r in rows} != expected or
            any(r['tokens'] != (1024 if r['chunk']==0 else 1023) for r in rows)):
        raise ValueError('incomplete private boundary coverage')
    first=next((dict(chunk=r['chunk'],layer=r['layer'],phase=r['phase'],label=r['label'])
                for r in rows if not r['bitwise_equal']),None)
    return dict(scope='synchronized selected layer2/layer3 snapshots; not performance qualification',
                records=rows,first_difference=first)
