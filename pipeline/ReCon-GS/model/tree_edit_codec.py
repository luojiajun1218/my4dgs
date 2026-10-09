"""Compact, replayable edits on the cached binary forest.

Swap and merge operations name existing nodes. A split stores one packed
membership bit per GS in that node, in materialized GS row order. No image,
gradient, floating coordinate, or full per-GS routing table is transmitted.
"""

import torch


MAGIC = b'NED1'
TAGS = {'swap': 1, 'merge': 2, 'split': 3}


def _write_integer(value):
    value = int(value)
    if value < 0:
        raise ValueError('Edit integers must be nonnegative')
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def encode_operation(operation):
    kind = operation['kind']
    if kind not in TAGS:
        raise ValueError('Unsupported tree operation')
    result = bytearray([TAGS[kind]])
    if kind in ('swap', 'merge'):
        keys = ('left', 'right') if kind == 'swap' else ('a', 'b')
        for key in keys:
            result.extend(_write_integer(operation[key]))
    else:
        labels = operation['labels']
        if not labels or any(label not in (0, 1) for label in labels) or len(set(labels)) != 2:
            raise ValueError('A split needs two nonempty binary child memberships')
        for value in (operation['level'], operation['node'], len(labels)):
            result.extend(_write_integer(value))
        packed = bytearray((len(labels) + 7) // 8)
        for index, label in enumerate(labels):
            packed[index // 8] |= int(label) << (index % 8)
        result.extend(packed)
    return bytes(result)


def encode_node_edits(operations):
    if not operations:
        return b''
    return MAGIC + _write_integer(len(operations)) + b''.join(encode_operation(op) for op in operations)


def decode_node_edits(payload):
    if not payload.startswith(MAGIC):
        raise ValueError('Unsupported node edit stream')
    cursor = len(MAGIC)
    def integer():
        nonlocal cursor
        value, shift = 0, 0
        while True:
            if cursor >= len(payload) or shift >= 64:
                raise ValueError('Truncated or oversized tree edit integer')
            symbol = payload[cursor]
            cursor += 1
            value |= (symbol & 127) << shift
            if symbol < 128:
                return value
            shift += 7
    count = integer()
    if count > len(payload) - cursor:
        raise ValueError('Tree operation count exceeds payload')
    operations = []
    for _ in range(count):
        if cursor >= len(payload):
            raise ValueError('Truncated tree operation')
        tag = payload[cursor]
        cursor += 1
        if tag in (1, 2):
            keys = ('left', 'right') if tag == 1 else ('a', 'b')
            operation = dict(kind='swap' if tag == 1 else 'merge')
            operation.update({key: integer() for key in keys})
        elif tag == 3:
            level, node, count = integer(), integer(), integer()
            size = (count + 7) // 8
            if count < 2 or cursor + size > len(payload):
                raise ValueError('Truncated split membership')
            packed = payload[cursor:cursor + size]
            cursor += size
            if count % 8 and packed[-1] >> (count % 8):
                raise ValueError('Nonzero split padding')
            labels = [(packed[index // 8] >> (index % 8)) & 1 for index in range(count)]
            if len(set(labels)) != 2:
                raise ValueError('Split has an empty child')
            operation = dict(kind='split', level=level, node=node, labels=labels)
        else:
            raise ValueError('Unknown tree operation tag')
        operations.append(operation)
    if cursor != len(payload):
        raise ValueError('Trailing tree edit bytes')
    return operations


def replay_node_edits(layout, operations):
    """Return edited index/counts/parents; the caller recreates pivots from XYZ."""
    index = layout['index'].detach().cpu().clone()
    counts = list(layout['counts'])
    parents = [value.detach().cpu().clone() for value in layout['parents']]
    seen = set()
    for operation in operations:
        kind = operation['kind']
        if kind == 'swap':
            left, right = operation['left'], operation['right']
            if len(counts) < 2 or left == right or not (0 <= left < counts[1] and 0 <= right < counts[1]):
                raise ValueError('Invalid subtree swap')
            a, b = int(parents[0][left]), int(parents[0][right])
            if a == b:
                raise ValueError('Swap must change a parent')
            parents[0][left], parents[0][right] = b, a
            index[0, index[1] == left] = b
            index[0, index[1] == right] = a
        elif kind == 'merge':
            a, b = operation['a'], operation['b']
            if len(counts) != 4 or a == b or not (0 <= a < counts[0] and 0 <= b < counts[0]):
                raise ValueError('Invalid common-parent merge')
            window = (index[0] == a) | (index[0] == b)
            if not bool((index[0] == a).any()) or not bool((index[0] == b).any()) or bool((index[3, window] >= 0).any()):
                raise ValueError('Merge requires two nonempty initial-depth branches')
            ca = torch.where(parents[0] == a)[0].tolist()
            cb = torch.where(parents[0] == b)[0].tolist()
            if len(ca) != 2 or len(cb) != 2:
                raise ValueError('Merge requires binary root children')
            fine_a = torch.where((index[2] >= 0) & (index[0] == a))[0]
            fine_b = torch.where((index[2] >= 0) & (index[0] == b))[0]
            slots = sorted(set(index[2, fine_a].tolist()))
            old_fine = sorted(set(index[2, window].tolist()) - {-1})
            if len(slots) != 4 or len(old_fine) != 8:
                raise ValueError('Merge does not have the expected terminal branches')
            old_coarse, old_leaf = index[1].clone(), index[2].clone()
            index[0, window] = a
            index[1, fine_a], index[1, fine_b] = ca[0], ca[1]
            for position, node in enumerate(ca + cb):
                index[2, old_coarse == node] = slots[position]
            for position, node in enumerate(old_fine):
                rows = (old_leaf == node) & window
                index[3, rows] = counts[3] + position
                parent = int(index[2, torch.where(rows)[0][0]])
                parents[2] = torch.cat((parents[2], parents[2].new_tensor([parent])))
            counts[3] += len(old_fine)
        elif kind == 'split':
            level, node = operation['level'], operation['node']
            if not 0 <= level < len(counts) - 1 or not 0 <= node < counts[level]:
                raise ValueError('Invalid split parent')
            if (level, node) in seen or bool((parents[level] == node).any()):
                raise ValueError('Split parent already has children')
            rows = torch.where(index[level] == node)[0]
            labels = torch.tensor(operation['labels'], dtype=torch.long)
            if labels.shape != rows.shape or len(set(labels.tolist())) != 2:
                raise ValueError('Split membership differs from cached members')
            index[level + 1, rows] = counts[level + 1] + labels
            counts[level + 1] += 2
            parents[level] = torch.cat((parents[level], parents[level].new_tensor([node, node])))
            seen.add((level, node))
        else:
            raise ValueError('Unsupported tree operation')
    return index, counts, parents
