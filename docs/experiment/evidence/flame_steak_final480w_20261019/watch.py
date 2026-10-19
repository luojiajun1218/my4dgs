"""Watch the authorized complete sequence, then verify every decoded frame."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

C=Path(__file__).resolve().parent
SOURCE=C/'source'
LAUNCH=json.loads((C/'launch.json').read_text())
RUN=Path(LAUNCH['run'])
STATE=C/'progress.json'
OUT=C/'measurement'

def running(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')

def save(value):
    value['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    temporary=STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(STATE)

def power():
    return subprocess.check_output(['nvidia-smi','--query-gpu=power.limit,power.draw,temperature.gpu','--format=csv,noheader'],text=True,timeout=10).strip()

def progress():
    rows=[]
    for path in sorted((RUN/'frame_summary').glob('frame_*.json')):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            pass
    assert [row['frame'] for row in rows]==list(range(1,len(rows)+1))
    milestones=[]
    for last in range(29,300,30):
        if len(rows)<last:
            continue
        first=max(1,last-29)
        part=rows[first-1:last]
        complete=[r['encoded_bytes']+(RUN/r['frame_manifest']).stat().st_size for r in part]
        milestones.append(dict(first_dynamic_frame=first,last_frame=last,
            psnr=statistics.mean(r['evaluation']['test']['psnr'] for r in part),
            complete_update_kb=statistics.mean(complete)/1000,
            train_wall_seconds=statistics.mean(r['elapsed_seconds'] for r in part),
            final_gaussians=part[-1]['total_count']))
    return dict(phase='training',training_pid=LAUNCH['pid'],run=str(RUN),completed_dynamic_frames=len(rows),last_completed_frame=rows[-1]['frame'] if rows else 0,milestones=milestones)

def main():
    while True:
        state=progress()
        save(state)
        with (C/'power_samples.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(),power=power()))+'\n')
        if not running(LAUNCH['pid']):
            break
        time.sleep(30)
    assert state['completed_dynamic_frames']==299
    assert 'Finished pipeline my4dgs.' in (C/'training.log').read_text(errors='replace')
    assert (RUN/'rigid-ckpt-299-300.pth').is_file()
    state['phase']='independent_sequence_benchmark'
    save(state)
    OUT.mkdir(exist_ok=False)
    benchmark=SOURCE/'scripts/benchmark_rigid_residual_tree.py'
    env={**os.environ,'PYTHONPATH':str(SOURCE)+os.pathsep+os.environ.get('PYTHONPATH',''),'MAX_JOBS':'4'}
    command=[sys.executable,str(benchmark),f'--run={RUN}','--sequence','--mode=stream','--sequence-warmups=10','--sequence-renders=50','--segment-size=30','--gpu-check-interval=30','--psnr-tolerance=0.00001','--ssim-tolerance=0.000001','--l1-tolerance=0.000001',f'--output={OUT/"decoded_sequence.json"}']
    before=power()
    assert before.split(',')[0].strip() == '480.00 W', before
    with (OUT/'benchmark.log').open('x') as f:
        subprocess.run(command,cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    provenance=dict(command=command,benchmark_sha256=hashlib.sha256(benchmark.read_bytes()).hexdigest(),pipeline_source=str(SOURCE),power_before=before,power_after=power())
    assert provenance['power_after'].split(',')[0].strip() == '480.00 W', provenance
    (OUT/'measurement_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    state['phase']='final_statistics'
    save(state)
    with (OUT/'postprocess.log').open('x') as f:
        subprocess.run([sys.executable,str(C/'postprocess_measurement.py'),f'--run={RUN}',f'--source={SOURCE}',f'--benchmark={OUT/"decoded_sequence.json"}'],cwd=SOURCE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    measured=json.loads((OUT/'summary.json').read_text())
    assert measured['frames']==300 and measured['dynamic_frames']==299 and measured['full_sequence_evaluation']
    state['phase']='complete'
    state['summary']=str(OUT/'summary.json')
    save(state)

if __name__=='__main__':
    try:
        main()
    except Exception:
        save(dict(phase='failed',error=traceback.format_exc()))
        raise
