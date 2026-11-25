"""Attribute A2 outputs with the frozen CLEAR evaluator, after online replay."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root, out, truth):
    repo = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo / "tools"))
    from diagnose_vttiny_botsort_m import clear_step, mot
    out.mkdir(parents=True,exist_ok=True)
    aud = read(root / "structure_audit.json")
    assert aud["status"] == "PASS"
    scores = {(v["tracker"], v["sequence"]): v for v in read(root / "metrics/summary.json")["results"]}
    seqs = read(root / "protocols/B_ref.json")["sequences"]
    by_seq, tracks, totals = [], [], {}
    for seq, n in seqs.items():
        gts = mot(truth / seq / "gt/gt.txt", n, True)
        frames_per_gt = sum(map(len, gts))
        reference = set()
        reference_ids = {}
        for name in ["B_ref"] + [k for k in aud["trackers"] if k != "B_ref"]:
            preds = mot(root/"trackers"/name/"data"/(seq+".txt"), n)
            weak = {e["track_id"]: e for e in aud["trackers"][name]["sequences"][seq]["weak_tracks"]}
            previous, last, coverage = {}, {}, set()
            hits, fps, peak = Counter(), Counter(), Counter()
            ids = switches_at_birth = total_fp = 0
            same_ref_id_at_endpoints = stable_ref_at_birth = 0
            last_frame = {}
            for frame, (gt, pred) in enumerate(zip(gts, preds), 1):
                pairs, geom = clear_step(gt, pred, previous)
                matched = set(pairs.values())
                total_fp += len(pred)-len(matched)
                for j, row in enumerate(pred):
                    tid = int(row[0])
                    if tid in weak:
                        peak[tid] = max(peak[tid], float(geom[:, j].max(initial=0)))
                        if j in matched: hits[tid] += 1
                        else: fps[tid] += 1
                for i, j in pairs.items():
                    gid, tid = int(gt[i,0]), int(pred[j,0])
                    coverage.add((frame, gid))
                    if name == "B_ref":
                        reference_ids[frame,gid] = tid
                    if gid in last and last[gid] != tid:
                        ids += 1
                        previous_ref = reference_ids.get((last_frame[gid],gid))
                        stable_ref = previous_ref is not None and previous_ref == reference_ids.get((frame,gid))
                        same_ref_id_at_endpoints += int(stable_ref)
                        if tid in weak and weak[tid]["birth_frame"] == frame:
                            switches_at_birth += 1
                            stable_ref_at_birth += int(stable_ref)
                    last[gid] = tid
                    last_frame[gid] = frame
            metric = scores[name, seq]
            assert (ids, total_fp, frames_per_gt-len(coverage)) == (metric["IDSW"], metric["FP"], metric["FN"]), (seq, name)
            if name == "B_ref": reference = coverage
            far = [tid for tid in weak if peak[tid] < .1]
            covered_ids={gid for _,gid in coverage}
            reference_gids={gid for _,gid in reference}
            row = dict(method=name, sequence=seq, gained_vs_B=len(coverage-reference),
                       lost_vs_B=len(reference-coverage), weak_births=len(weak),
                       additional_gt_ids=len(covered_ids-reference_gids),
                       lost_gt_ids=len(reference_gids-covered_ids),
                       switches_B_kept_id_at_endpoints=same_ref_id_at_endpoints,
                       weak_birth_switches_B_kept_id_at_endpoints=stable_ref_at_birth,
                       switches_at_weak_birth=switches_at_birth, never_near_ids=len(far),
                       never_near_fp_frames=sum(fps[k] for k in far), weak_fp_frames=sum(fps.values()))
            totals.setdefault(name, Counter()).update({k:v for k,v in row.items() if isinstance(v,int)})
            by_seq.append(row)
            for tid,e in weak.items():
                assert hits[tid]+fps[tid] == e["output_frames"]
                tracks.append(dict(method=name, sequence=seq, track_id=tid, birth_frame=e["birth_frame"],
                                   confirmation_delay=e["confirmation_delay_frames"], tp_frames=hits[tid],
                                   fp_frames=fps[tid], max_gt_iou=peak[tid]))
    output = []
    for name, counts in totals.items():
        delays=[e["confirmation_delay"] for e in tracks if e["method"]==name]
        output.append(dict(method=name, **counts, median_delay=float(np.median(delays)) if delays else None,
                           p90_delay=float(np.percentile(delays,90)) if delays else None))
    write_csv(out/"summary.csv",output)
    write_csv(out/"per_sequence.csv",by_seq)
    write_csv(out/"weak_tracks.csv",tracks)
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=Path(__file__).resolve().parent/"full30")
    parser.add_argument("--gt-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    main(args.run_dir,args.output,args.gt_root)
