"""Resume immutable completed forecasts after runner timeout. Model code is unchanged."""
import argparse, hashlib, inspect, json, os
from pathlib import Path
import audit_driver  # Retains the verified exchange-timezone boundary fix.
import audit_300_24h as audit

ap=argparse.ArgumentParser()
ap.add_argument('--raw',default='evidence/TSLA_1h_extended_raw.csv')
ap.add_argument('--model',choices=['small'],default='small')
ap.add_argument('--shard',type=int,required=True)
ap.add_argument('--out',default='results')
args=ap.parse_args()
assert args.shard in [0,1,3]
source_path=Path(audit.__file__)
source_bytes=source_path.read_bytes()
blob=hashlib.sha1(f'blob {len(source_bytes)}\0'.encode()+source_bytes).hexdigest()
assert blob=='cbdc2eb8e24cb2fe2e0944deb377791b1f1138e2', 'Original benchmark source changed'
f=Path(args.out)/f'forecasts_{args.model}_{args.shard}.jsonl'
assert f.exists(), 'Missing persisted partial artifact'
initial_sha=hashlib.sha256(f.read_bytes()).hexdigest()
lines=f.read_text().splitlines();existing=[];truncated_tail=False
for j,line in enumerate(lines):
    try:r=json.loads(line)
    except json.JSONDecodeError:
        if j!=len(lines)-1:raise
        truncated_tail=True
        continue
    assert r['model']==args.model and r['origin_index']%6==args.shard
    assert len(r['predicted_ohlc'])==len(r['future_times'])==17
    existing.append(r)
done={r['origin_index'] for r in existing}
assert len(done)==len(existing) and len(existing)<=50
expected=set(range(args.shard,300,6))
assert done<=expected
if truncated_tail:
    f.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in existing))
print('RECOVERY_PLAN',json.dumps({'shard':args.shard,'retained':len(existing),'missing':sorted(expected-done),'initial_sha256':initial_sha}),flush=True)
# Three audited scheduling/persistence substitutions only. All numerical inference
# statements, timestamps, seeds, history, samples, and weights remain verbatim.
src=inspect.getsource(audit.run)
changes=[
    ('results=[];chosen=[(i,r) for i,r in enumerate(origins) if i%6==args.shard];',
     'results=list(_recovery_existing);chosen=[(i,r) for i,r in enumerate(origins) if i%6==args.shard and i not in _recovery_done];'),
    ("with f.open('w') as handle:","with f.open('a') as handle:"),
    ("'n_requested':len(chosen)","'n_requested':50,'n_reused':len(_recovery_existing),'n_newly_computed':len(chosen),'runtime_scope':'resume_only'")
]
for old,new in changes:
    assert src.count(old)==1, 'Unexpected source format during controlled resume patch'
    src=src.replace(old,new,1)
audit.__dict__['_recovery_existing']=existing
audit.__dict__['_recovery_done']=done
exec(compile(src,'<audited_persistence_resume>','exec'),audit.__dict__)
audit.run(args)
final=[json.loads(s) for s in f.read_text().splitlines()]
assert len(final)==50 and {r['origin_index'] for r in final}==expected
final_map={r['origin_index']:r for r in final}
assert all(final_map[r['origin_index']]==r for r in existing), 'A completed prediction was altered'
receipt={'initial_run':35495599954,'recovery_run':os.environ.get('GITHUB_RUN_ID'),'execution_commit':os.environ.get('GITHUB_SHA'),'shard':args.shard,'n_retained_unmodified':len(existing),'n_newly_computed':50-len(existing),'n_final':50,'initial_file_sha256':initial_sha,'final_file_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'truncated_incomplete_tail_removed':truncated_tail,'all_original_complete_rows_unchanged':True,'numerical_model_code_changed':False}
(Path(args.out)/f'recovery_receipt_{args.shard}.json').write_text(json.dumps(receipt,indent=2))
print('RECOVERY_COMPLETE='+json.dumps(receipt),flush=True)
