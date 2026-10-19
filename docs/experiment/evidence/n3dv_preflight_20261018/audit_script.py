from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image
from plyfile import PlyData

BASE=Path('/home/gamer-02/Documents/DeepLearning/data/n3dv')
OUT=Path('/home/gamer-02/Documents/DeepLearning/my4dgs/output/n3dv_preprocessing_audit_20261018')
SCENES=('coffee_martini','cook_spinach','cut_roasted_beef','flame_salmon_1','flame_steak','sear_steak')
spec=importlib.util.spec_from_file_location('existing_n3dv_validator','/home/gamer-02/Documents/DeepLearning/S2GS/scripts/pre_dynerf/validate_dataset.py')
validator=importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def verify_png(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return None
    except Exception as error:
        return dict(path=str(path),error=str(error))


def inspect_scene(name):
    started=time.perf_counter()
    root=BASE/(name+'_result')
    result=validator.validate_dataset(root,300,BASE/name)
    result['name']=name
    pngs=sorted(path for camera in result['cameras'] for path in (root/camera/'images').glob('*.png'))
    with ThreadPoolExecutor(max_workers=8) as pool:
        failures=[error for error in pool.map(verify_png,pngs) if error is not None]
    result.update(png_crc_checked=len(pngs),png_crc_failures=failures,
        test_camera='cam00',train_cameras=len(result['cameras'])-1,
        missing_camera_ids_from_raw=[f'cam{i:02d}' for i in range(21) if f'cam{i:02d}' not in result['cameras']])
    poses=np.load(root/'poses_bounds.npy')
    frames=poses[:,:15].reshape(-1,3,5)
    rotations=np.stack((frames[:,:,1],frames[:,:,0],-frames[:,:,2]),axis=-1)
    centers=frames[:,:,3]
    extent=float(1.1*np.linalg.norm(centers[1:]-centers[1:].mean(0),axis=1).max())
    vertices=PlyData.read(root/'points3D_downsample2.ply')['vertex'].data
    xyz=np.stack([vertices[k] for k in ('x','y','z')],axis=-1)
    colors=np.stack([vertices[k] for k in ('red','green','blue')],axis=-1).astype(np.float64)/255
    assert np.isfinite(xyz).all() and np.isfinite(colors).all()
    assert np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))<1e-6
    assert np.max(np.abs(np.linalg.det(rotations)-1))<1e-6
    radius=np.linalg.norm(xyz-np.median(xyz,axis=0),axis=1)
    result['geometry']=dict(points=len(xyz),unique_points=len(np.unique(xyz,axis=0)),
        finite=True,orthogonality_max_error=float(np.max(np.abs(rotations.transpose(0,2,1)@rotations-np.eye(3)))),
        determinant_min=float(np.linalg.det(rotations).min()),camera_extent=extent,
        point_radius_quantiles=dict(zip(('p50','p99','p999','max'),np.quantile(radius,[.5,.99,.999,1]).tolist())),
        far_points_over_10_camera_extents=int((radius>10*extent).sum()),
        extreme_to_median_radius=float(radius.max()/np.median(radius)),
        focal_range=[float(frames[:,2,4].min()),float(frames[:,2,4].max())],
        calibrated_dimensions=[float(frames[0,1,4]),float(frames[0,0,4])])
    photo=[]
    sample_loaded=0
    for index,camera in enumerate(result['cameras']):
        samples=[]
        for frame in (0,149,299):
            with Image.open(root/camera/'images'/f'{frame:04d}.png') as image:
                samples.append(np.asarray(image.convert('RGB')).copy())
            sample_loaded+=1
        assert any(np.any(samples[0]!=image) for image in samples[1:]),camera
        image=samples[0]
        h,w=image.shape[:2]
        # Match the existing S2GS loader's width-scaled scalar focal length.
        focal=frames[0,2,4]*w/frames[0,1,4]
        position=(xyz-centers[index])@rotations[index]
        z=position[:,2]
        pixels=position[:,:2]/np.maximum(z[:,None],1e-12)*focal+np.array([w/2,h/2])
        good=(z>0)&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        ids=np.where(good)[0]
        ids=ids[np.argsort(z[ids],kind='stable')]
        grid=np.floor(pixels[ids]/4).astype(np.int64)
        key=grid[:,1]*((w+3)//4)+grid[:,0]
        _,chosen=np.unique(key,return_index=True)
        ids=ids[chosen]
        uv=pixels[ids].astype(int)
        error=np.mean(np.abs(image[uv[:,1],uv[:,0]].astype(float)/255-colors[ids]),axis=1)
        photo.append(dict(camera=camera,points_in_front=int((z>0).sum()),points_projecting_inside=int(good.sum()),
            foreground_grid_samples=len(ids),median_color_L1=float(np.median(error)),
            p90_color_L1=float(np.quantile(error,.9)),
            point_depth_above_10x_far=int((z>10*poses[index,-1]).sum())))
    result['decoded_sample_images']=sample_loaded
    result['projection_diagnostics']=photo
    result['projection_scope']='Sparse 4px z-buffer RGB consistency under the existing S2GS loader. Diagnostic only; not feature reprojection error or proof of geometric accuracy.'
    model=root/'frame000001/sparse/0'
    result['retained_original_colmap_model']=all((model/f'{part}.bin').is_file() for part in ('cameras','images','points3D'))
    result['retained_fullres_undistorted_300']=all((root/f'frame{i+1:06d}/images/cam00.png').is_file() for i in range(300))
    log=root/'preprocess.log'
    if log.exists():
        text=log.read_text(errors='replace')
        fused=re.findall(r'Number of fused points:\s*(\d+)',text)
        result['dense_fused_points_reported']=int(fused[-1]) if fused else None
        result['farthest_sampling_reported']='Farthest point sampled points: 40000' in text
    result['source_metadata_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('poses_bounds.npy','points3D_downsample2.ply')}
    rawvideo=BASE/name/'cam00.mp4'
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(rawvideo)],check=True,text=True,capture_output=True)
    result['raw_cam00_video']=json.loads(probe.stdout)['streams'][0]
    result['elapsed_seconds']=time.perf_counter()-started
    print(json.dumps(dict(scene=name,images=result['images'],crc_failures=len(failures),
        camera_extent=extent,far_points=result['geometry']['far_points_over_10_camera_extents'],
        maximum_radius=float(radius.max()),sample_decodes=sample_loaded,seconds=result['elapsed_seconds'])),flush=True)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name in SCENES:
        rows.append(inspect_scene(name))
        (OUT/'audit.partial.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(schema='n3dv_readonly_preflight_v1',root=str(BASE),scenes=rows,
        total_pngs=sum(row['images'] for row in rows),
        all_png_crc_passed=not any(row['png_crc_failures'] for row in rows),
        pointclouds_finite=True,training_started=False,source_data_modified=False,
        caveat='Five processed scenes retain LLFF poses and downsampled PLY but have deleted original COLMAP/dense intermediates. Structural checks and color-projection diagnostics do not certify full SfM/MVS accuracy.')
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Complete readonly audit:',OUT/'audit.json',flush=True)


if __name__=='__main__':
    main()
