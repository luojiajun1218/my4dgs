"""Lossless zero-subtree control, not a learned structure selection method.

Diagnostic only; never imported by the training pipeline. Uses existing RDP1.
"""
from dataclasses import replace
import importlib
from pathlib import Path
import sys
import struct

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def compact_packet(packet, tree):
    """Validate and losslessly recode an existing packet; no encoder latent.

    The temporary Projection's distortion/objective are placeholders, never
    reported: a packet alone does not contain its original fitting target.
    """
    field = P.decode_projection(packet, tree)
    _, channels, count = struct.unpack('<III', packet[4:16])
    header = 32 + 8 * channels
    step = np.frombuffer(packet[32:header], dtype='<f8').copy()
    bits = np.unpackbits(np.frombuffer(packet[header:], dtype=np.uint8))[:count]
    position = 0

    def bit():
        nonlocal position
        value = int(bits[position])
        position += 1
        return value

    def symbol():
        zeros = 0
        while bit() == 0:
            zeros += 1
        number = 1
        for _ in range(zeros):
            number = 2 * number + bit()
        mapped = number - 1
        return (mapped + 1) // 2 if mapped % 2 else -(mapped // 2)

    root = np.array([symbol() for _ in range(channels)], dtype=np.int64)
    details = {}

    def visit(index):
        node = tree.nodes[index]
        if node.left >= 0 and bit():
            details[index] = np.array([symbol() for _ in range(channels)], dtype=np.int64)
            visit(node.left)
            visit(node.right)

    visit(0)
    if position != count:
        raise AssertionError('Validated packet was not consumed exactly')
    original = P.Projection(field, root, details, tuple(details), step, count, 0., 0.)
    if P.encode_projection(original, tree) != packet:
        raise AssertionError('Parsed symbols do not reproduce the source packet')
    compact = compact_zero_subtrees(original, tree, 0.)
    recoded = P.encode_projection(compact, tree)
    return recoded, dict(
        original_packet_bytes=len(packet), compacted_packet_bytes=len(recoded),
        original_payload_bits=count, compacted_payload_bits=compact.payload_bits,
        original_splits=len(details), compacted_splits=len(compact.split_nodes),
        decoded_field_exact=True)


def compact_zero_subtrees(projection, tree, rate_weight):
    """Remove a split iff it and every encoded descendant have zero symbols.

    Zero ancestors of nonzero descendants remain, including their actual bits.
    The supplied rate weight is used only to update the reported objective.
    """
    if not np.isfinite(rate_weight) or rate_weight < 0:
        raise ValueError('Invalid rate weight')
    needed = set()

    def retain(index):
        if index not in projection.detail_symbols:
            return False
        node = tree.nodes[index]
        if node.left < 0:
            raise ValueError('Detail attached to atomic leaf')
        left = retain(node.left)
        right = retain(node.right)
        keep = left or right or bool(np.any(projection.detail_symbols[index]))
        if keep:
            needed.add(index)
        return keep

    retain(0)
    details = {i: q.copy() for i, q in projection.detail_symbols.items() if i in needed}
    bits = sum(P.signed_symbol_bits(q) for q in projection.root_symbols)

    def count(index):
        node = tree.nodes[index]
        if node.left < 0:
            return 0
        if index not in details:
            return 1
        return (1 + sum(P.signed_symbol_bits(q) for q in details[index])
                + count(node.left) + count(node.right))

    bits += count(0)
    field = P.reconstruct(tree, projection.root_symbols, details, projection.quant_step)
    if not np.array_equal(field, projection.field):
        raise AssertionError('Zero compaction changed decoded field')
    result = replace(projection, field=field, detail_symbols=details,
                     split_nodes=tuple(details), payload_bits=bits,
                     objective=projection.distortion + rate_weight * bits)
    packet = P.encode_projection(result, tree)
    if not np.array_equal(P.decode_projection(packet, tree), projection.field):
        raise AssertionError('Compacted packet changed decoded field')
    if bits > projection.payload_bits:
        raise AssertionError('Compaction increased the payload')
    return result


def main():
    import argparse
    from datetime import datetime
    import hashlib
    import json
    import zipfile
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path, nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    output = root / 'output' / ('idea_zero_subtree_control_'
                               + datetime.now().strftime('%Y%m%dT%H%M%S_%f'))
    output.mkdir(exist_ok=False)
    sources = [Path(__file__), root / 'pipeline/ReCon-GS/model/rd_tree_motion.py',
               root / 'pipeline/ReCon-GS/model/rd_tree_projection.py']
    source_hashes = {}
    with zipfile.ZipFile(output / 'sources.zip', 'x') as archive:
        for path in sources:
            data = path.read_bytes()
            name = str(path.relative_to(root))
            source_hashes[name] = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
    M = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_motion')
    records = []
    for index, path in enumerate(args.checkpoints):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint['deformation']
        motion = M.RDTreeMotion(M.RDTreeConfig(**state['config']))
        motion.restore(state)  # Recomputed original packet must match checkpoint.
        original = motion.project()
        compact = compact_zero_subtrees(original, motion.tree, motion.rate)
        packet = P.encode_projection(compact, motion.tree)
        packet_name = f'compacted_{index:03d}.rdp'
        with (output / packet_name).open('xb') as stream:
            stream.write(packet)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        row = dict(checkpoint=str(path.resolve()), checkpoint_sha256=digest,
                   original_packet_bytes=len(state['packet']),
                   compacted_packet_bytes=len(packet),
                   original_payload_bits=original.payload_bits,
                   compacted_payload_bits=compact.payload_bits,
                   original_splits=len(original.split_nodes),
                   compacted_splits=len(compact.split_nodes),
                   decoded_field_exact=True, packet=packet_name)
        records.append(row)
        print(json.dumps(row), flush=True)
    report = dict(scope='Checkpoint-local lossless zero-subtree control. No retraining, '
                  'no image-quality improvement claimed, no full-sequence recoding claim. '
                  'Diagnostic packets require the corresponding shared candidate tree. '
                  'Protocol and incremental bytes are unchanged and excluded here.',
                  source_sha256=source_hashes, records=records)
    with (output / 'report.json').open('x') as stream:
        json.dump(report, stream, indent=2)
    print('Saved:', output, flush=True)


if __name__ == '__main__':
    main()
