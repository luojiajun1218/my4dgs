"""Export hierarchy capacity counts for a later content-blind control, not a codec."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = {}
    for frame in range(1, 300):
        path = args.run / 'frame_summary' / f'frame_{frame:03d}.json'
        assert path.exists(), 'Capacity control budget requires a completed299-frame treatment'
        packet = json.loads(path.read_text())
        assert 'wall' in packet['timing']
        activity = packet['stream']['activities']
        rows[str(frame)] = dict(level_counts=[
            [level['counts'][action] for action in ('KEEP', 'SHARED', 'SPLIT')]
            for level in activity['levels']],
            observed_local_GS_count=activity['gs']['counts']['SPLIT'],
            observed_local_GS_fraction=activity['gs_local_active_fraction'],
            auxiliary_graph_budget=14, observed_extra_graphs=packet['stream']['candidate_fit']['extra_renders'])
    packet = dict(frames=rows, source_run=str(args.run), completed299=True,
        semantics='Counts only; no learned node IDs, cameras, images or test-quality selection.',
        limits='Same reachable action counts alone do not ensure exact activeGS-local capacity: region occupancy can differ in the control. Allocated/full and effective capacity must be reported separately.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2))
    print(json.dumps(dict(frames=299,output=str(args.output))))


if __name__ == '__main__':
    main()
