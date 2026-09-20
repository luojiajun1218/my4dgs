"""Deterministic lossless control; no renderer or dataset needed."""
import importlib
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from zero_subtree_compaction import compact_zero_subtrees, compact_packet

P = importlib.import_module('pipeline.ReCon-GS.model.rd_tree_projection')


def test_zero_subtree_compaction():
    xyz = np.column_stack((np.arange(8), np.zeros((8, 2))))
    tree = P.build_candidate_tree(xyz, max_depth=3, min_leaf_size=1)
    for target in (np.zeros((8, 1)),
                   np.array([1., -1., 1., -1., 0., 0., 0., 0.])[:, None]):
        source = P.project_rd_tree(target, tree, np.array([1.]), 0.,
                                   fixed_cut_depth=3)
        result = compact_zero_subtrees(source, tree, .01)
        torch.testing.assert_close(torch.from_numpy(result.field),
                                   torch.from_numpy(source.field), rtol=0, atol=0)
        assert result.payload_bits < source.payload_bits
        assert len(result.split_nodes) < len(source.split_nodes)
        again = compact_zero_subtrees(result, tree, .01)
        assert P.encode_projection(again, tree) == P.encode_projection(result, tree)
        original_packet = P.encode_projection(source, tree)
        recoded, stats = compact_packet(original_packet, tree)
        assert recoded == P.encode_projection(result, tree)
        assert stats['compacted_packet_bytes'] == len(recoded)
        assert stats['decoded_field_exact']
        try:
            compact_packet(original_packet[:-1], tree)
        except ValueError:
            pass
        else:
            raise AssertionError('Truncated packet accepted')
        assert result.objective == source.distortion + .01 * result.payload_bits
        if np.any(target):
            assert 0 in result.detail_symbols
            assert not np.any(result.detail_symbols[0])
            assert any(np.any(q) for q in result.detail_symbols.values())
        else:
            assert not result.split_nodes
            assert result.payload_bits == 2  # Root zero symbol and stop flag.

    rng = np.random.default_rng(1307)
    target = rng.normal(size=(8, 6)) * .1
    target[4:] = 0.
    for vectorized in (False, True):
        source = P.project_rd_tree(target, tree, np.full(6, .01), .001,
                                   fixed_cut_depth=3, vectorized_symbols=vectorized,
                                   vectorized_statistics=vectorized)
        original = P.encode_projection(source, tree)
        recoded, stats = compact_packet(original, tree)
        torch.testing.assert_close(torch.from_numpy(P.decode_projection(recoded, tree)),
                                   torch.from_numpy(source.field), rtol=0, atol=0)
        assert stats['compacted_packet_bytes'] == len(recoded) <= len(original)
        assert compact_packet(recoded, tree)[0] == recoded
        adaptive = P.project_rd_tree(target, tree, np.full(6, .01), .001,
                                     vectorized_symbols=vectorized,
                                     vectorized_statistics=vectorized)
        adaptive_packet = P.encode_projection(adaptive, tree)
        # Positive-rate DP must already remove strictly redundant zero subtrees.
        assert compact_packet(adaptive_packet, tree)[0] == adaptive_packet


if __name__ == '__main__':
    test_zero_subtree_compaction()
