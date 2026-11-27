"""Verify published B/GMC evidence against its checksums and frozen evaluation protocols."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_json(path):
    return json.loads(path.read_bytes())


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    code = Path(__file__).resolve().parent
    repo = code.parents[2]
    archive = repo/'results/vttiny/iterations/v0.4-motion-blur/gmc/04-b-gmc-recovery'
    manifest = read_json(archive/'MANIFEST.json')['files']
    for name, expected in manifest.items():
        path = repo/name
        assert path.stat().st_size==expected['size'] and digest(path)==expected['sha256'], name

    paths = ('joint/full/metrics','forensics/full','pending/full/metrics')
    audits = [read_json(archive/p/'evaluation_audit.json') for p in paths]
    joint, forensic, pending = [audit['validation']['protocols'] for audit in audits]
    frozen = joint['native']
    assert len(frozen['sequences']) == 30 and sum(frozen['sequences'].values()) == 12590
    fields = ('sequences','detector_sha256','rgb_encoder_sha256','detection_fingerprints',
              'feature_hashes','decoded_image_hashes','native_config','boxmot_source_hashes')
    for protocols in (joint,forensic,pending):
        for method, protocol in protocols.items():
            assert all(protocol[k]==frozen[k] for k in fields), method
    for audit in audits:
        assert audit['ground_truth']['sha256'] == audits[0]['ground_truth']['sha256']
        assert audit['evaluator']['commit'] == '12c8791b303e0a0b50f753af204249e622d0281a'
    for method in joint:
        assert joint[method]['output_prediction_hashes']==forensic[method]['output_prediction_hashes']
    before = forensic['evidence_both']['output_prediction_hashes']
    assert before==pending['evidence_both']['output_prediction_hashes']
    after = pending['pending_guarded']['output_prediction_hashes']
    assert {s for s in before if before[s]!=after[s]} == {'DJI_0030_2','DJI_0169_3','DJI_0221_2'}

    guarded = pending['pending_guarded']['pending_identity']
    assert guarded['source']==digest(code/'pending/pending.py')
    assert guarded['refined_source']==digest(code/'pending/pending_refined.py')
    assert forensic['evidence_both']['lost_camera_policy']['source_sha256']==digest(code/'forensics/lost_camera_margin.py')
    focus = read_json(archive/'pending/focus2/parity.json')
    assert focus['pending_shadow']['focus_sha256']==before['DJI_0030_2']
    assert focus['pending_guarded']['focus_sha256']==after['DJI_0030_2']

    outcomes = read_csv(archive/'pending/full/guarded_release_context.csv')
    assert Counter(row['outcome'] for row in outcomes)=={'same_gt':3,'FP':2}
    assert all(not row['reference_prior_gt_owner_at_release'] for row in outcomes if row['outcome']=='same_gt')
    assert any((r['seq'],r['frame'],r['tid'],r['current_gt'])==('DJI_0030_2','416','7','4') for r in outcomes)
    old = read_csv(archive/'pending/DJI_0030_2_历史67个误报逐项复核.csv')
    assert len({(r['frame'],r['det']) for r in old})==len(old)==67
    assert all(r['pending_guarded']=='absent' for r in old)
    assert sum(r['pending_temporal']=='FP' for r in old)==51
    print(f'PASS: {len(manifest)} files, 30 sequences, 12,590 frames, 5 guarded releases')


if __name__ == '__main__':
    main()
