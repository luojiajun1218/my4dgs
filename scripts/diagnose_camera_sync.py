"""Read-only temporal/epipolar consistency probe, not a synchronization correction.

Compare dynamic-area SIFT correspondences across candidate times. Calibration
and photometric ambiguities remain; reported minima alone do not prove offsets.
"""
import argparse
import json
from pathlib import Path
import sys
import math
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from pipeline.data import DataConfig, DataModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--query-camera', default='cam13')
    parser.add_argument('--reference-cameras', nargs='+', default=['cam12', 'cam14'])
    parser.add_argument('--query-frames', nargs='+', type=int, default=[0, 60, 120])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cv2.setNumThreads(4)
    data = DataModule(DataConfig(root=str(args.root), resolution=1,
                                extra_dataset_kwargs={'num_frames': 1}, shuffle=False))
    data.setup()
    cameras = {item.path.stem: item for item in data.scene.train_dataset.items}
    cameras.update({item.path.stem: item for item in data.scene.test_dataset.items})

    def fundamental(a, b):
        def calibration(item):
            intrinsic = np.array([[item.width/(2*math.tan(item.fovx/2)), 0, item.width/2],
                [0, item.height/(2*math.tan(item.fovy/2)), item.height/2], [0, 0, 1]])
            return intrinsic, item.R.T, item.T
        ka, ra, ta = calibration(cameras[a])
        kb, rb, tb = calibration(cameras[b])
        rotation = rb@ra.T
        tx, ty, tz = tb-rotation@ta
        skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
        return np.linalg.inv(kb).T@skew@rotation@np.linalg.inv(ka)

    def image(camera, frame):
        value = cv2.imread(str(args.root/'frames'/f'{frame:04d}'/f'{camera}.png'), cv2.IMREAD_GRAYSCALE)
        if value is None:
            raise FileNotFoundError((camera, frame))
        return value

    masks = {}
    for camera in [args.query_camera, *args.reference_cameras]:
        samples = np.stack([image(camera, frame) for frame in [0, 29, 90, 149, 200, 299]])
        mask = (samples.astype(np.float32).std(0) > 10).astype(np.uint8)*255
        h, w = mask.shape
        rectangle = np.zeros_like(mask)
        rectangle[int(h*.22):int(h*.80), int(w*.32):int(w*.81)] = 255
        mask = cv2.dilate(mask, np.ones((11, 11), np.uint8)) & rectangle
        masks[camera] = mask
        cv2.imwrite(str(args.output/f'{camera}_dynamic_mask.png'), mask)
    sift = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    cache = {}

    def features(camera, frame):
        if (camera, frame) not in cache:
            cache[camera, frame] = sift.detectAndCompute(image(camera, frame), masks[camera])
        return cache[camera, frame]

    curves = []
    for reference in args.reference_cameras:
        f = fundamental(args.query_camera, reference)
        for query_frame in args.query_frames:
            qa, da = features(args.query_camera, query_frame)
            curve = []
            for frame in range(max(0, query_frame-80), min(299, query_frame+160)+1, 5):
                qb, db = features(reference, frame)
                if da is None or db is None or min(len(da), len(db)) < 2:
                    continue
                reverse = {m.queryIdx: m.trainIdx for m, n in matcher.knnMatch(db, da, k=2)
                           if m.distance < .75*n.distance}
                pairs = [m for m, n in matcher.knnMatch(da, db, k=2)
                         if m.distance < .75*n.distance and reverse.get(m.trainIdx) == m.queryIdx]
                if not pairs:
                    continue
                a = np.array([(*qa[m.queryIdx].pt, 1) for m in pairs])
                b = np.array([(*qb[m.trainIdx].pt, 1) for m in pairs])
                fa, ftb = a@f.T, b@f
                residual = (b*fa).sum(1)
                sampson = np.abs(residual)/np.sqrt((fa[:, :2]**2).sum(1)+(ftb[:, :2]**2).sum(1))
                curve.append(dict(reference_frame=frame, delta=frame-query_frame,
                    matches=len(pairs), median_sampson_px=float(np.median(sampson)),
                    inlier_fraction_1px=float((sampson < 1).mean())))
            eligible = [row for row in curve if row['matches'] >= 12]
            best = min(eligible, key=lambda row: row['median_sampson_px']) if eligible else None
            zero = next((row for row in curve if row['delta'] == 0), None)
            report = dict(query_camera=args.query_camera, reference_camera=reference,
                          query_frame=query_frame, best=best, zero_delta=zero, curve=curve)
            curves.append(report)
            (args.output/'epipolar_curves.json').write_text(json.dumps(dict(
                scope='Dynamic ROI SIFT probe; not a verified time correction', curves=curves), indent=2))
            print(json.dumps({key: report[key] for key in
                              ('reference_camera', 'query_frame', 'best', 'zero_delta')}), flush=True)


if __name__ == '__main__':
    main()
