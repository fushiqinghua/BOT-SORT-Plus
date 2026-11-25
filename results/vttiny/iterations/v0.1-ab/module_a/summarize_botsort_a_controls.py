"""Check paired control inputs and archive the diagnostic counts and local case evidence."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from render_botsort_a_cases import iou, read_mot


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(path):
    with path.open(encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def main(exp, diag):
    output = diag / "analysis"
    ref = read(exp / "protocols/A.json")
    original = read(exp / "metrics/comparison.json")
    controls = read(diag / "controls/metrics_all/comparison.json")
    matched_keys = ["detector_sha256", "rgb_encoder_sha256", "source_protocol_sha256", "sequences",
                    "detection_fingerprints", "feature_hashes", "native_config", "boxmot_source_hashes"]
    names = [e["tracker"] for e in controls["results"]]
    seqs = ref["sequences"]
    checked = {}
    for n in ["baseline", "A"] + names:
        root = exp if n in ("baseline", "A") else diag / "controls"
        proto = read(root / "protocols" / (n+".json"))
        assert proto["status"] == "COMPLETE" and all(ref[k] == proto[k] for k in matched_keys), n
        for seq in seqs:
            raw = (root / "trackers" / n / "data" / (seq+".txt")).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == proto["output_prediction_hashes"][seq]
        events = Counter()
        for seq in seqs:
            events.update(read(root / "timing" / n / (seq+".json"))["event_counts"])
        checked[n] = dict(weak_birth=events["weak_birth"], suppressed_native_birth=events["suppressed_native_birth"])
    byte_diff = {}
    for n, orig in (("audit_A","A"),("empty_pool_A","baseline"),("shadow_A","baseline")):
        byte_diff[n] = [seq for seq in seqs if
                        (diag/"controls/trackers"/n/"data"/(seq+".txt")).read_bytes() !=
                        (exp/"trackers"/orig/"data"/(seq+".txt")).read_bytes()]
    assert not byte_diff["audit_A"]
    subset = diag/"subset_check"
    s = read(subset/"protocols/native_lost_pool_check.json")
    assert s["status"] == "COMPLETE" and set(s["sequences"]) == set(byte_diff["empty_pool_A"])
    residual = [seq for seq in s["sequences"] if
                (subset/"trackers/native_lost_pool_check/data"/(seq+".txt")).read_bytes() !=
                (exp/"trackers/baseline/data"/(seq+".txt")).read_bytes()]
    metrics = ["HOTA","DetA","AssA","IDF1","MOTA","FP","FN","IDSW"]
    rows = [dict(tracker=e["tracker"], **{k:e[k] for k in metrics}, **checked[e["tracker"]])
            for e in original["results"][:2] + controls["results"]]
    with (output/"metrics_with_controls.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    gt = {(r["sequence"],r["gt"]):r for r in table(output/"gt_tracks.csv")}
    sw = table(output/"switches.csv")
    p = table(output/"promotions.csv")
    novel = [e for e in sw if gt[e["sequence"],e["gt"]]["baseline_matched"] == "0"]
    at_birth = [e for e in sw if e["origin"] == "weak" and e["birth_now"] == "True"]
    old_in_lost = [float(e["previous_lost_iou"]) for e in at_birth if e["previous_id_in_lost_at_weak_birth"] == "True"]
    never_near = [e for e in p if 0 <= float(e["lifetime_max_iou"]) < .1]
    no_output = [e for e in p if int(e["output_frames"]) == 0]
    a_counts = read(output/"attribution.json")["counts"]
    gts = a_counts["paired_gt"]
    assert sum(gts.values())==179772 and checked["A"]["weak_birth"]==len(p)
    assert len(no_output)==sum(e["native_kept"]=="False" for e in p)
    assert all(float(e["lost_max_iou"])>.85 for e in no_output)
    src = a_counts["fp_geometry"]
    fp_net = {k: src.get("weak|"+k,0) + src.get("native|"+k,0) - src.get("baseline|"+k,0)
              for k in ("under_10","10_30","30_50","ge_50")}
    seq = "DJI_0327_3"
    seq_gt = read(diag/"controls/metrics_all/evaluation_audit.json")["ground_truth"]["root"]
    data = [read_mot(Path(seq_gt)/seq/"gt/gt.txt"), read_mot(exp/"trackers/baseline/data"/(seq+".txt")),
            read_mot(exp/"trackers/A/data"/(seq+".txt"))]
    fp = data[2][data[2][:,1] == 212]
    assert np.array_equal(fp[:,0],np.arange(111,219))
    maximum = [max(float(iou(row[2:6], arr[arr[:,0]==row[0],2:6]).max(initial=0)) for row in fp)
               for arr in data[:2]]
    evidence = dict(input_keys_identical=matched_keys, sequences=len(seqs), frames=sum(seqs.values()),
                    prediction_files_sha256_verified=len(seqs)*len(checked), weak_birth_and_native_veto=checked,
                    nonidentical_prediction_files=byte_diff, native_lost_rule_subset=list(s["sequences"]),
                    native_lost_rule_subset_still_different=residual,
                    net_fp_by_max_gt_iou=fp_net, weak_birth_switches=len(at_birth), previous_id_still_lost=len(old_in_lost),
                    prior_lost_iou_50_to_85=sum(.5 <= v <= .85 for v in old_in_lost),
                    prior_lost_iou_under_50=sum(v < .5 for v in old_in_lost),
                    switches_on_gt_never_seen_by_baseline=len(novel),
                    same_baseline_id_at_both_observations=sum(e["baseline_same_id_at_endpoints"]=="True" for e in sw),
                    gt_with_more_switches=sum(int(e["A_switches"])>int(e["baseline_switches"]) for e in gt.values()),
                    no_output=len(no_output), no_output_all_removed_by_duplicate_check=True,
                    no_output_all_last_window_hits_same_gt=sum(e["gt_support"]=="same_gt_all" for e in no_output),
                    never_near_ids=len(never_near), never_near_output_frames=sum(int(e["fp_frames"]) for e in never_near),
                    confirmation_window_entirely_iou_below_10=sum(e["raw_all_iou_lt10"]=="True" for e in never_near),
                    missing_smooth_rgb_at_weak_birth=sum(e["smooth_rgb_at_birth"]=="False" for e in p),
                    sustained_fp_case=dict(sequence=seq, A_id=212,frames=[111,218], count=len(fp),
                                           max_overlap_GT=maximum[0],max_overlap_baseline=maximum[1]))
    (output/"verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for key in ("exp","diag"):
        parser.add_argument("--"+key,type=Path,required=True)
    main(**vars(parser.parse_args()))
