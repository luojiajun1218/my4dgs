"""Train-only tracked SfM triangles -> image-verified surface supports.

No original SfM xyz, distorted database coordinates, cam00 images, BA or SGD.
The work cap is explicit; an untested triangle is never called unsupported.
"""
from pathlib import Path
import json,struct,time
import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation

NAMES=tuple(f'cam{i:02d}.png' for i in range(1,21))

def colmap_half_pixel_coordinates(xy,full_size,half_size):
 """COLMAP corner convention -> native/CV pixel-centre index convention."""
 return np.asarray(xy)*np.asarray(half_size,dtype=np.float64)/np.asarray(full_size)-.5

def _array_hwc(value):
 if hasattr(value,'detach'):value=value.detach().cpu().numpy()
 result=np.asarray(value)
 # remap otherwise recopies a transposed full image for every local support.
 return np.ascontiguousarray(result.transpose(1,2,0) if result.ndim==3 and result.shape[0]==3 else result)

def _residual_score(residual,xy):
 if not len(xy):return 0.
 return float(np.square(_sample(np.asarray(residual,dtype=np.float32),xy)).sum())

def _select(proposals,active_uids,budget):
 ordered=sorted(proposals,key=lambda x:(-x['score'],x['uid']))
 bindings=[x for x in ordered if x['uid'] in active_uids]
 fresh=[x for x in ordered if x['uid'] not in active_uids][:budget]
 return fresh+bindings,len(fresh),len(bindings)

def summary(v):
 a=np.asarray(v,dtype=np.float64).reshape(-1)
 if not len(a):return dict(count=0)
 return dict(count=len(a),min=float(a.min()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()),mean=float(a.mean()))

def _tracks(path,train_ids):
 """Read IDs/observations; SKIP xyz/rgb/error bytes rather than parsing xyz."""
 result=[];ambiguous=0
 with open(path,'rb') as f:
  count=struct.unpack('<Q',f.read(8))[0]
  for _ in range(count):
   uid=struct.unpack('<Q',f.read(8))[0];f.seek(35,1) # 3 doubles xyz + RGB3 + double error
   n=struct.unpack('<Q',f.read(8))[0]
   pairs=np.frombuffer(f.read(n*8),dtype='<i4').reshape(n,2)
   selected=pairs[np.isin(pairs[:,0],list(train_ids))]
   if len(np.unique(selected[:,0]))!=len(selected):ambiguous+=1;continue
   if len(selected)>=3:result.append((uid,selected.copy()))
 return result,dict(total_sfm_points=count,ambiguous_train_tracks_skipped=ambiguous,unambiguous_ge3_train_tracks=len(result),original_xyz_used=False)

def _project(x,p,k):
 q=np.asarray(x)@p[:,:3].T+p[:,3]
 return q[:,:2]/q[:,2:3]*np.array([k[0,0],k[1,1]])+k[:2,2],q[:,2]

def _triangulate(obs,ps,ks):
 a=[]
 for cam,xy in obs.items():
  uv=np.linalg.solve(ks[cam],np.r_[xy,1.]);p=ps[cam]
  a.extend((uv[0]*p[2]-p[0],uv[1]*p[2]-p[1]))
 _,_,vh=np.linalg.svd(a,full_matrices=False);h=vh[-1]
 if abs(h[3])<1e-12:return None
 x=h[:3]/h[3];errors=[];js=[];rays=[]
 for cam,xy in obs.items():
  p,k=ps[cam],ks[cam];q=p[:,:3]@x+p[:,3]
  if q[2]<=0:return None
  projected,_=_project(x[None],p,k);errors.append(np.linalg.norm(projected[0]-xy))
  j=np.array([[k[0,0]/q[2],0,-k[0,0]*q[0]/q[2]**2],[0,k[1,1]/q[2],-k[1,1]*q[1]/q[2]**2]])@p[:,:3];js.append(j)
  centre=-p[:,:3].T@p[:,3];d=x-centre;rays.append(d/np.linalg.norm(d))
 rms=float(np.sqrt(np.mean(np.square(errors))));angle=float(np.arccos(np.clip((np.asarray(rays)@np.asarray(rays).T).min(),-1,1))*180/np.pi)
 if rms>1. or max(errors)>2. or angle<2.:return None
 j=np.concatenate(js);cov=np.linalg.pinv(j.T@j)*max(.25,rms)**2
 return x,cov,rms,angle

def _sample(image,xy):
 return cv2.remap(image,xy[:,0].astype(np.float32)[:,None],xy[:,1].astype(np.float32)[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)[:,0]

def propose_tracked_surface_births(frame_index,train_views,sfm_path,budget,active_uids=(),residuals=None,*,image_root):
 """Unique future-module API. Specs are fresh geometry, no old ownership.

 train_views=None reads only cam01..20 half images. residuals optionally maps
 those names to half RGB residuals, for support ranking; absent = supply audit.
 active_uids prevent duplicate births, while current binding specs are reported.
 """
 start=time.perf_counter();sfm_path=Path(sfm_path)
 if isinstance(budget,bool) or not isinstance(budget,int) or budget<0:raise ValueError('budget must be a nonnegative candidate count')
 if train_views is not None and set(train_views)!=set(NAMES):raise ValueError('Exactly the20 standard training cameras are required, without cam00')
 active_uids=set(active_uids)
 residual_arrays=None if residuals is None else {name:_array_hwc(residuals[name]) for name in NAMES}
 from libgs.data.utils.colmap import read_extrinsics_binary,read_intrinsics_binary,qvec2rotmat
 all_images=read_extrinsics_binary(sfm_path/'images.bin');cams=read_intrinsics_binary(sfm_path/'cameras.bin')
 image_root=Path(image_root)
 images={i:im for i,im in all_images.items() if Path(im.name).name in NAMES}
 assert len(images)==20
 records,report=_tracks(sfm_path/'points3D.bin',images)
 ps={};ks={};gray0={};gray={};rgb={};percam={};observations={uid:{} for uid,_ in records}
 read_names=[]
 for cam,im in sorted(images.items()):
  name=Path(im.name).name;c=cams[im.camera_id]
  if c.model!='PINHOLE' or len(c.params)!=4:raise ValueError('Undistorted PINHOLE SfM is required; do not pass distorted database coordinates')
  fx,fy,cx,cy=c.params
  ps[cam]=np.column_stack((qvec2rotmat(im.qvec),im.tvec))
  ref=cv2.imread(str(image_root/'0000'/name),cv2.IMREAD_COLOR);read_names.append(f'0000/{name}')
  if train_views is None:
   current=cv2.imread(str(image_root/f'{int(frame_index):04d}'/name),cv2.IMREAD_COLOR);read_names.append(f'{int(frame_index):04d}/{name}')
  else:
   value=train_views[name].image
   if hasattr(value,'detach'):value=value.detach().cpu().numpy()
   current=np.asarray(value).transpose(1,2,0)[:,:,::-1]
   current=np.uint8(np.rint(np.clip(current*255,0,255)))
  if ref is None or current is None:raise ValueError(f'Missing training image {name}')
  if current.shape[:2]!=ref.shape[:2]:raise ValueError('All supervision must use matching half-resolution images')
  height,width=ref.shape[:2];scale=np.array([width/c.width,height/c.height])
  ks[cam]=np.array([[fx*scale[0],0,cx*scale[0]-.5],[0,fy*scale[1],cy*scale[1]-.5],[0,0,1.]])
  gray0[cam]=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY);gray[cam]=cv2.cvtColor(current,cv2.COLOR_BGR2GRAY)
  rgb[cam]=current[:,:,::-1].astype(np.float32)/255
  members=[(uid,int(pairs[pairs[:,0]==cam,1][0])) for uid,pairs in records if bool((pairs[:,0]==cam).any())]
  if not members:
   percam[name]=dict(initial=0,LK_forward_backward_pass=0);continue
  initial=np.asarray([colmap_half_pixel_coordinates(im.xys[j],(c.width,c.height),(width,height)) for _,j in members],np.float32).reshape(-1,1,2)
  moved,ok,err=cv2.calcOpticalFlowPyrLK(gray0[cam],gray[cam],initial,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  reverse,back,_=cv2.calcOpticalFlowPyrLK(gray[cam],gray0[cam],moved,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
  fb=np.linalg.norm(reverse[:,0]-initial[:,0],axis=1);xy=moved[:,0]
  valid=ok[:,0].astype(bool)&back[:,0].astype(bool)&np.isfinite(xy).all(1)&(fb<=.75)&(xy[:,0]>=2)&(xy[:,0]<width-2)&(xy[:,1]>=2)&(xy[:,1]<height-2)
  for index,(uid,_) in enumerate(members):
   if valid[index]:observations[uid][cam]=xy[index].astype(np.float64)
  percam[name]=dict(initial=len(members),LK_forward_backward_pass=int(valid.sum()),FB_pixels=summary(fb[valid]),temporal_displacement_pixels=summary(np.linalg.norm(xy[valid]-initial[:,0][valid],axis=1)))
 tracks=[];obs_counts=[];failed=0
 for uid,obs in observations.items():
  if len(obs)<3:continue
  result=_triangulate(obs,ps,ks)
  if result is None:failed+=1;continue
  x,cov,error,angle=result;tracks.append(dict(uid=int(uid),xyz=x,cov=cov,obs=obs,reprojection=error,parallax=angle));obs_counts.append(len(obs))
 report.update(frame=int(frame_index),per_camera=percam,tracked_ge3=sum(len(x)>=3 for x in observations.values()),geometry_failed=failed,triangulated_tracks=len(tracks),track_camera_count=summary(obs_counts),track_reprojection_RMS_half_pixels=summary([x['reprojection'] for x in tracks]),parallax_deg=summary([x['parallax'] for x in tracks]),tracking_seconds=time.perf_counter()-start)
 # Delaunay triangle identity is based on original track IDs, never SfM xyz.
 triangles={}
 for cam in images:
  ids=[i for i,t in enumerate(tracks) if cam in t['obs']]
  if len(ids)<3:continue
  xy=np.stack([tracks[i]['obs'][cam] for i in ids])
  for face in Delaunay(xy).simplices:
   key=tuple(sorted(ids[int(i)] for i in face))
   if key in triangles:continue
   common=set.intersection(*(set(tracks[i]['obs']) for i in key))
   if len(common)<3:continue
   areas={v:abs(np.linalg.det(np.column_stack((tracks[key[1]]['obs'][v]-tracks[key[0]]['obs'][v],tracks[key[2]]['obs'][v]-tracks[key[0]]['obs'][v]))))*.5 for v in common}
   ref=max(areas,key=areas.get);area=areas[ref]
   if not 16<=area<=4096:continue
   triangles[key]=dict(common=common,ref=ref,area=area)
 def triangle_uid(key):return 'tri:'+':'.join(map(str,sorted(tracks[i]['uid'] for i in key)))+':0'
 ordered=sorted(triangles.items(),key=lambda x:(triangle_uid(x[0]) not in active_uids,-len(x[1]['common']),-x[1]['area'],x[0]))
 work_cap=max(512,4*int(budget),len(active_uids));funnel=dict(candidate_triangles=len(ordered),work_cap=work_cap,geometry_stable=0,photo_threeview=0,connected_support=0,surface_thickness=0)
 proposals=[];nccs=[];all_support=[]
 for key,info in ordered[:work_cap]:
  tri=np.stack([tracks[i]['xyz'] for i in key]);normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]);norm=np.linalg.norm(normal)
  if norm<=1e-12:continue
  normal/=norm;edge=max(np.linalg.norm(tri[(i+1)%3]-tri[i]) for i in range(3));altitude=norm/edge
  uncertainty=np.sqrt(np.median([normal@tracks[i]['cov']@normal for i in key]))
  if uncertainty>altitude*.25:continue
  funnel['geometry_stable']+=1;ref=info['ref'];corners=np.stack([tracks[i]['obs'][ref] for i in key])
  low=np.ceil(corners.min(0)).astype(int);high=np.floor(corners.max(0)).astype(int)
  xx,yy=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1));pixels=np.column_stack((xx.ravel(),yy.ravel()))
  if not len(pixels):continue
  mat=np.column_stack((corners[1]-corners[0],corners[2]-corners[0]))
  if abs(np.linalg.det(mat))<1e-9:continue
  uv=(pixels-corners[0])@np.linalg.inv(mat).T;bary=np.column_stack((1-uv.sum(1),uv));inside=(bary>=0).all(1)
  pixels=pixels[inside];bary=bary[inside]
  if len(pixels)<16:continue
  z=tri@ps[ref][2,:3]+ps[ref][2,3];weights=bary/z[None];world=(weights@tri)/weights.sum(1)[:,None]
  refcolour=_sample(rgb[ref],pixels);refgray=refcolour.mean(1);centred=refgray-refgray.mean()
  if np.square(centred).mean()<1e-4:continue
  samples=[refcolour];sample_valids=[np.ones(len(pixels),dtype=bool)];sample_xy=[pixels];valid_counts=np.ones(len(pixels),dtype=np.int32);good_views=[ref]
  for cam in sorted(info['common']-{ref}):
   xy,depth=_project(world,ps[cam],ks[cam]);colour=_sample(rgb[cam],xy);v=(depth>0)&(xy[:,0]>=1)&(xy[:,0]<rgb[cam].shape[1]-1)&(xy[:,1]>=1)&(xy[:,1]<rgb[cam].shape[0]-1)
   if not v.all():continue
   cg=colour.mean(1);other=cg-cg.mean();ncc=float(np.dot(centred,other)/np.sqrt(np.dot(centred,centred)*np.dot(other,other)+1e-20));nccs.append(ncc)
   if ncc<.9:continue
   photo_valid=np.max(np.abs(colour-refcolour),axis=1)<=.08
   valid_counts+=photo_valid;samples.append(colour);sample_valids.append(photo_valid);sample_xy.append(xy);good_views.append(cam)
  if len(good_views)<3:continue
  funnel['photo_threeview']+=1;valid=valid_counts>=3
  mask=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8);mask[pixels[:,1]-low[1],pixels[:,0]-low[0]]=valid
  count,labels,component,_=cv2.connectedComponentsWithStats(mask,8)
  if count<2:continue
  largest=1+np.argmax(component[1:,cv2.CC_STAT_AREA]);chosen=labels[pixels[:,1]-low[1],pixels[:,0]-low[0]]==largest
  if chosen.sum()<16:continue
  funnel['connected_support']+=1
  binary=np.uint8(labels==largest)
  distance=cv2.distanceTransform(cv2.copyMakeBorder(binary,1,1,1,1,cv2.BORDER_CONSTANT,value=0),cv2.DIST_L2,5)[1:-1,1:-1]
  pick=np.argmax(distance[pixels[:,1]-low[1],pixels[:,0]-low[0]]);mu=world[pick]
  selected_world=world[chosen];t0=(tri[1]-tri[0]);t0/=np.linalg.norm(t0);t1=np.cross(normal,t0);basis=np.column_stack((t0,t1))
  coords=(selected_world-mu)@basis;cov=coords.T@coords/len(coords)
  eigen,axes=np.linalg.eigh(cov)
  if eigen[0]<=1e-12:continue
  # Three-sigma ellipse is bounded by the actual connected support boundary,
  # not the triangle bbox/old source covariance. Holes remain boundaries.
  eroded=cv2.erode(binary,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0);boundary=((labels==largest)&(eroded==0))[pixels[:,1]-low[1],pixels[:,0]-low[0]]&chosen
  d2=np.einsum('ni,ij,nj->n',coords[np.flatnonzero(boundary[chosen])],np.linalg.inv(cov),coords[np.flatnonzero(boundary[chosen])]) if boundary.any() else np.array([0.])
  factor=np.sqrt(max(0,float(d2.min())))/3
  sigmas=np.sqrt(eigen)*factor
  if min(sigmas)<=uncertainty or min(sigmas)<=0:continue
  funnel['surface_thickness']+=1;rotation=np.column_stack((basis@axes,normal))
  if np.linalg.det(rotation)<0:rotation[:,0]*=-1
  q=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
  colours=np.stack(samples)[:,chosen];photo=np.stack(sample_valids)[:,chosen]
  reliable_colours=np.where(photo[:,:,None],colours,np.nan)
  support_colour=np.nanmedian(reliable_colours,axis=0);colour=np.median(support_colour,axis=0)
  ids=sorted(tracks[i]['uid'] for i in key);uid='tri:'+':'.join(map(str,ids))+':0'
  score=float(chosen.sum()*len(good_views))
  if residual_arrays is not None:
   score=sum(_residual_score(residual_arrays[Path(images[v].name).name],sample_xy[j][chosen&sample_valids[j]]) for j,v in enumerate(good_views))/20
  raw=dict(xyz=mu.astype(np.float32),rotation=q.astype(np.float32),logscale=np.log(np.r_[sigmas,uncertainty]).astype(np.float32),
    opacity=np.array([-2.1972246],np.float32),sh_dc=((colour-.5)/.28209479177387814)[None].astype(np.float32),sh_rest=np.zeros((3,3),np.float32))
  proposals.append(dict(uid=uid,binding_only=uid in active_uids,track_ids=ids,born_frame=int(frame_index),raw=raw,camera_bits=sum(1<<(int(Path(images[v].name).stem[3:])-1) for v in good_views),
   support_pixels=pixels[chosen].astype(np.float32),support_world=selected_world.astype(np.float32),support_color=support_colour.astype(np.float32),
   reference_camera=Path(images[ref].name).name,score=score,normal=normal.astype(np.float32),triangulation_cov=np.mean([tracks[i]['cov'] for i in key],0),
   support_diagnostics=dict(verified_pixels=int(chosen.sum()),triangle_reference_area=float(info['area']),photo_views=len(good_views),sigma_world=np.r_[sigmas,uncertainty].tolist(),
     footprint='3sigma ellipse inside verified connected support, not actual rendered Talpha area'),
   provenance='train-only LK observations -> current DLT triangle -> cross-view image warp; no original SfM xyz'))
  all_support.append(int(chosen.sum()))
 result,fresh_count,binding_count=_select(proposals,active_uids,budget)
 report.update(triangle_funnel=funnel,triangles_tested=min(len(ordered),work_cap),untested_due_work_cap=max(0,len(ordered)-work_cap),
  photo_NCC=summary(nccs),reliable_support_specs=len(proposals),returned_fresh=fresh_count,active_rebindings=binding_count,active_binding_only_geometry=True,active_missing_keep_old_geometry=sorted(active_uids-{x['uid'] for x in result if x['binding_only']}),support_pixel_count=summary(all_support),
  returned_support_pixels=sum(x['support_diagnostics']['verified_pixels'] for x in result if not x['binding_only']),read_image_names=read_names,active_rebindings_preserve_original_age=True,
  external_test_images_read=False,DB_used=False,SGD_steps=0,BA_steps=0,pixel_center_transform='half_native_xy = full_undistorted_xy*[halfW/fullW,halfH/fullH]-.5',ranking='actual residual SSE over verified support' if residuals is not None else 'geometry-supply audit only; production caller must supply residuals',seconds=time.perf_counter()-start,
  limitations=['SfM calibrations/track associations are shared provided inputs; original xyz and cam00 observations are never used for current DLT.',
   'Support is discrete verified image-plane evidence, not native visible contribution. Occlusion/material ambiguity remains; module must measure actual Talpha/codec.',
   ('Score is sampled current residual SSE on photo-verified support, not native Talpha-weighted error or achievable gain.' if residuals is not None else 'No rendered residual supplied in standalone supply audit: score is verified support times photo-view count, not reconstruction-error benefit.'),
   'Opacity .1 is an explicit candidate placeholder; no physical birth, native density fit, training or acceptance occurred.'])
 return result,report
