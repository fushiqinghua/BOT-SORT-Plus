"""Replay and verify the four B-centered low-update ablations on the server."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from low_updates import compose_low_update, read, save, sha


ARMS = {'b_off_check':(False,False), 'b_bam_check':(True,False),
        'b_low_r':(False,True), 'b_bam_low_r':(True,True)}
SEQUENCES = {'1_rgb_val_1-08':600, '2_rgb_val_2-08':600}
FROZEN = ('native_config','sequences','detector_sha256','rgb_encoder_sha256','feature_hashes',
          'decoded_image_hashes','detection_fingerprints','encoder_contracts','boxmot_source_hashes',
          'source_protocol_sha256','implementation_sha256')


def roots(project):
    return (project/'m3ot_first_batch_v1',
            project/'m3ot_second_batch_v1/tracking/bam',
            project.parent/'VT_Tiny_MOT_Baseline_v1/integrations')


def run_arm(args):
    first,_,integrations=roots(args.project)
    frozen=first/'tracking/run/higher'
    protocol=read(frozen/'protocols/B_early_moderate.json')
    asset=first/'inputs/higher'
    if protocol['status']!='COMPLETE' or read(asset/'ready.json')['status']!='verified':
        raise ValueError('Original M3OT B inputs have not been verified')
    source=integrations/'b_gmc_joint_r1/repo'
    if any(sha(source/k)!=v for k,v in protocol['implementation_sha256'].items()):
        raise ValueError('Archived B runner source has changed')
    sys.path[:0]=list(map(str,(source,source/'tools',args.project/'tracking_runtime')))
    import cv2
    import torch
    import botsort_plus.native_adapter as adapter
    import run_botsort_plus as runner

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    cfg_path=args.output/'configs'/(args.arm+'.json')
    config=dict(protocol['config'],name=args.arm)
    save(cfg_path,config)
    prepare=runner.prepare
    bam,low_r=ARMS[args.arm]

    def configure(opts):
        cfg,paths,rec=prepare(opts)
        pre=[k for k in FROZEN if k not in ('decoded_image_hashes','boxmot_source_hashes')]
        if rec['config']!=config or any(rec[k]!=protocol[k] for k in pre):
            raise ValueError('B settings or frozen cache differ in the new arm')
        adapter.BoTSORTPlus=compose_low_update(adapter.BoTSORTPlus,args.project,args.output,
                                              args.arm,paths,rec,bam=bam,low_r=low_r)
        return cfg,paths,rec

    runner.prepare=configure
    params=['--config',str(cfg_path),'--baseline-root',str(asset),'--features-dir',str(asset/'features'),
            '--images-root',str(args.project/'data/raw/M3OT'),'--runtime-root',str(args.project),
            '--output-dir',str(args.output),'--trace']
    return runner.run(runner.parser().parse_args(params))


def verify(args):
    first,bam,_=roots(args.project)
    frozen=first/'tracking/run/higher'
    baseline=read(frozen/'protocols/B_early_moderate.json')
    first_receipt=read(frozen/'frozen_inputs.json')
    bam_receipt=read(bam/'receipt.json')
    result={}
    hashes={}
    for arm,(enable_bam,enable_r) in ARMS.items():
        rec=read(args.output/'protocols'/(arm+'.json'))
        if rec['status']!='COMPLETE' or any(rec[k]!=baseline[k] for k in FROZEN):
            raise ValueError(f'Incorrect run scope or cache in {arm}')
        result[arm]={}
        hashes[arm]={}
        for seq,frames in SEQUENCES.items():
            track=args.output/'trackers'/arm/'data'/(seq+'.txt')
            trace=args.output/'events'/arm/(seq+'.jsonl')
            events=read(args.output/'low_update_events'/arm/(seq+'.json'))
            timing=read(args.output/'timing'/arm/(seq+'.json'))
            digest=sha(track)
            if timing['frames']!=frames or digest!=rec['output_prediction_hashes'][seq]:
                raise ValueError(f'Trajectory is incomplete: {arm} {seq}')
            raw_sof=first_receipt['action_counts']['B_early_moderate'][seq]['native_matrix_sha256']
            if events['native_sof_sha256']!=raw_sof or events['frames']!=frames:
                raise ValueError(f'Original SOF stream differs: {arm} {seq}')
            if not enable_r:
                prior_arm='b_bam' if enable_bam else 'B_early_moderate'
                prior=bam if enable_bam else frozen
                if sha(prior/'trackers'/prior_arm/'data'/track.name)!=digest or sha(
                        prior/'events'/prior_arm/trace.name)!=sha(trace):
                    raise ValueError(f'Individual control fails byte parity: {arm} {seq}')
                events['control_track_and_events_byte_equal']=True
            if enable_r and events['r_updates']!=events['accepted_low']:
                raise ValueError('R used a different scope from actual accepted lows')
            if enable_bam and enable_r and events['both_updates']!=events['bam_updates']:
                raise ValueError('Combined updates did not apply both mechanisms')
            hashes[arm][seq]=digest
            result[arm][seq]=events
    save(args.output/'receipt.json',dict(status='COMPLETE',sequences=SEQUENCES,
         input_proof={k:baseline[k] for k in FROZEN},
         output_prediction_hashes=hashes,events=result,
         contracts={arm:read(args.output/'protocols'/(arm+'.json'))['low_updates'] for arm in ARMS}))


def dispatch(args):
    jobs={}
    for arm in ARMS:
        env=dict(os.environ,YOLO_CONFIG_DIR=str(args.output/'runtime_config'/arm))
        with (args.output/(arm+'.log')).open('x',encoding='utf-8') as log:
            jobs[arm]=subprocess.Popen([sys.executable,'-Bu',str(Path(__file__)),
                         '--project',str(args.project),'--output',str(args.output),'--arm',arm],
                         env=env,stdout=log,stderr=subprocess.STDOUT)
    failed=[arm for arm,p in jobs.items() if p.wait()!=0]
    if failed:
        raise RuntimeError('Unfinished replay arms: '+', '.join(failed))
    verify(args)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=tuple(ARMS)+('all','verify'),default='all')
    a=p.parse_args()
    if a.arm=='all':
        dispatch(a)
    elif a.arm=='verify':
        verify(a)
    else:
        sys.exit(run_arm(a))
