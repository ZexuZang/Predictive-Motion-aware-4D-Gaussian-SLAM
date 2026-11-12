"""Read-only evidence audit: do not confuse GenFlow inference with actual GS updates."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re


def audit(log_path: Path, packet_dir: Path, result_dir: Path | None = None):
    if not log_path.is_file():raise FileNotFoundError(log_path)
    if not packet_dir.is_dir():raise FileNotFoundError(packet_dir)
    text=log_path.read_text(errors='replace')
    p=packet_dir/'runtime_diagnostics.jsonl'
    lines=p.read_text().splitlines() if p.is_file() else []
    records=[json.loads(s) for s in lines if s.strip()]
    idx=[int(d['frame']) for d in records if 'frame' in d]
    applied=[int(s) for s in re.findall(r'PREDICTIVE4D_GENFLOW_APPLIED frame=(\d+)',text)]
    fallback=[int(s) for s in re.findall(r'PREDICTIVE4D_NATIVE_FLOW_FALLBACK frame=(\d+)',text)]
    a={
       'raw_inference_records':len(records),
       'unique_inference_frames':len(set(idx)),
       'repeat_inference_records':len(idx)-len(set(idx)),
       'unique_genflow_gs_update_keyframes':len(set(applied)),
       'genflow_gs_update_events':len(applied),
       'raft_fallback_events':len(fallback),
       'unique_raft_fallback_keyframes':len(set(fallback)),
       'prediction_source':'official_GenFlow_rec_seq (recorded; validate weights independently)',
       'dynamic_mask_source':'native RAFT/YOLO prior; NOT learned GenFlow posterior',
       'uncertainty_calibrated':False,
    }
    a['genflow_application_fraction_among_logged_keyframes']=(
       len(set(applied)) / len(set(applied+fallback)) if (applied or fallback) else None)
    if result_dir is not None:
        a['full_trajectory_exists']=(result_dir/'plot/trj_final.json').is_file()
        for name in ('before_opt','after_opt'):
            a[f'rendering_{name}_exists']=(result_dir/f'psnr/{name}/final_result.json').is_file()
        if a['full_trajectory_exists']:
            tr=json.loads((result_dir/'plot/trj_final.json').read_text())
            a['n_trajectory_frames']=len(tr['trj_id'])
    return a


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--packet-dir',type=Path,required=True)
    p.add_argument('--result-dir',type=Path)
    p.add_argument('--save-json',type=Path)
    args=p.parse_args()
    result=audit(args.log,args.packet_dir,args.result_dir)
    data=json.dumps(result,indent=2)
    print(data)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True,exist_ok=True)
        args.save_json.write_text(data+'\n')
