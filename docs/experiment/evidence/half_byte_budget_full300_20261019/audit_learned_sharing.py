"""Read-only recoding at fixed integer motion; no gate decisions for training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.source.resolve()))
import torch
from pipeline.model.rigid_residual_tree import RigidTreeConfig
from pipeline.model.spatial_gate import SpatialGateConfig
from pipeline.model.spatial_gated_motion import SpatialGatedMotion

torch.set_num_threads(2)
if args.output.exists():
    raise FileExistsError(args.output)
args.output.mkdir(parents=True)
before=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
packet=checkpoint['rigid_tree_motion']
settings=copy.deepcopy(packet['config'])
settings['spatial_gate']=SpatialGateConfig(**settings['spatial_gate'])
config=RigidTreeConfig(**settings)

def load():
    model=SpatialGatedMotion(copy.deepcopy(config))
    model.restore(copy.deepcopy(packet))
    return model

actual=load()
assert actual._hard_gate_frozen
assert checkpoint['global_step'] == actual.motion_steps, 'Use the prebirth motion-phase checkpoint'
full=tuple(v.to(torch.int32).clone() for v in actual._full_motion_symbols())
original_xyz,original_q=actual.materialized_pose()
rows=[]

@torch.no_grad()
def record(name,model):
    current=model._full_motion_symbols()
    assert all(torch.equal(a,b.to(torch.int32)) for a,b in zip(full,current))
    xyz,q=model.materialized_pose()
    error=float((xyz-original_xyz).abs().max())
    model.register_materialized_layout(xyz,q)
    info=model.export_motion(args.output/(name+'.bin'))
    rows.append(dict(representation=name,packet_bytes=info['bytes'],selected_regions=int(model.owner.unique().numel()),gs_translation_blocks=int(model._values(hard=True)[2].ne(0).any(-1).sum()),integer_total_motion_unchanged=True,max_float_pose_roundoff=error))

record('actual_learned_hierarchy',actual)
coarse=load()
with torch.no_grad():
    coarse.opened.zero_()
    coarse.opened[0]=True
    coarse._recode_owner(coarse.spatial_paths[:,0],full_motion=full)
record('same_motion_existing_eight_coarse',coarse)
with torch.no_grad():
    coarse._refresh_active_representatives()
record('same_motion_rate_l1_refitted_eight_coarse',coarse)
individual=load()
with torch.no_grad():
    individual.opened.zero_()
    individual.opened[0]=True
    individual.region_translation.zero_()
    individual.region_rotation.zero_()
    individual._recode_owner(individual.spatial_paths[:,0],full_motion=full)
record('same_motion_all_gs_without_shared_predictor',individual)
result=dict(frame=checkpoint['current_frame'],step=checkpoint['global_step'],checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=before,checkpoint_unchanged=(before==hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()),scope='Prebirth motion packets only. Same quantized total motion and reference, with production encode/decode checks. Fitted coarse predictors minimize the existing rate/L1 proxy; this is representation diagnosis, not a retrained ablation or exact entropy optimum.',rows=rows)
assert result['checkpoint_unchanged']
(args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
