"""Read-only comparison of completed fixed-depth runs with historical best."""
import argparse,json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--best',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists() or a.output.is_symlink():raise FileExistsError(a.output)
summary=json.loads(a.summary.read_text());best=json.loads(a.best.read_text());results=[]
for scene,groups in summary['scenes'].items():
 historic=best.get(scene,{})
 for mode in ('depth2','depth3','depth4'):
  row=groups[mode];valid=historic.get('dynamic_frames',0)>=299 and historic.get('last_complete_frame',-1)>=299
  psnr=historic.get('dynamic_psnr');byte=historic.get('complete_update_bytes')
  comparable=bool(valid and isinstance(psnr,(int,float)) and isinstance(byte,(int,float)))
  dominates=bool(comparable and row['psnr']>psnr and row['dynamic_kb']*1000<=byte)
  results.append(dict(scene=scene,mode=mode,run=row['run'],comparable=comparable,
    dominates_historical_psnr_and_bytes=dominates,psnr=row['psnr'],dynamic_kb=row['dynamic_kb'],
    historical_psnr=psnr,historical_dynamic_kb=byte/1000 if isinstance(byte,(int,float)) else None,
    historical_availability=historic.get('availability')))
a.output.write_text(json.dumps(dict(results=results,dominant_candidates=[x for x in results if x['dominates_historical_psnr_and_bytes']]),indent=2)+'\n')
print('Dominant candidates:',sum(x['dominates_historical_psnr_and_bytes'] for x in results))
