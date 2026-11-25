"""Cross-check event keys and scope for the two D3 follow-up groups."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

FIRST_FAILURES = "round1_findings/D3_complete_first_failure_candidates.csv"
WINDOWS = "analysis/recovery_windows.csv"
B = ("aligned_but_outside_score", "aligned_and_eligible")
EXPECTED = {B[0]: 89, B[1]: 22, "no_aligned_box": 95, "no_detector_rows": 4}


def verified_rows(archive, member, files):
    content = archive.read(member)
    assert hashlib.sha256(content).hexdigest() == files[member], member
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))


def event_key(row):
    return row["sequence"], int(row["recovery_frame"]), int(row["track_id"])


def csv_write(name, rows):
    with Path(name).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    archive_path = Path(sys.argv[1])
    with archive_path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("EVIDENCE_MANIFEST.json"))["files"]
        failed = verified_rows(archive, FIRST_FAILURES, manifest)
        observable = [r for r in verified_rows(archive, WINDOWS, manifest)
                      if r["horizon"] == "10" and r["identity_status"] == "same_gt"
                      and r["complete_observable_window"] == "True"]
    window = {event_key(r):r for r in observable}
    assert len(window) == len(observable) == 287
    assert sum(r["stable_all_frames"] == "True" for r in observable) == 77
    assert {event_key(r) for r in failed} == {event_key(r) for r in observable if r["first_failure_offset"]}
    assert len({event_key(r) for r in failed}) == len(failed) == 210
    assert Counter(r["candidate_type"] for r in failed) == EXPECTED
    by_sequence = defaultdict(Counter)
    by_offset = defaultdict(Counter)
    focused = defaultdict(list)
    for r in failed:
        event = window[event_key(r)]
        offset = int(event["first_failure_offset"])
        assert 1 <= offset <= 10
        assert int(r["first_failure_frame"]) == int(r["recovery_frame"])+offset
        assert event["first_failure_reason"] == r["first_failure_reason"]
        assert int(r["gt_id"]) == int(event["gt_id"])
        by_sequence[r["sequence"]][r["candidate_type"]] += 1
        by_offset[offset][r["candidate_type"]] += 1
        if r["candidate_type"] in B:
            focused[r["candidate_type"]].append(r)
    one = {k:{"events": len(v), "recording_segments": len({r["sequence"] for r in v}),
              "recording_prefixes": len({r["sequence"].rsplit("_",1)[0] for r in v}),
              "distinct_gt_frames": len({(r["sequence"],r["first_failure_frame"],r["gt_id"]) for r in v}),
              "distinct_track_ids_by_sequence": len({(r["sequence"],r["track_id"]) for r in v}),
              "offset_first_3": sum(by_offset[i][k] for i in range(1,4)),
              "offset_first_5": sum(by_offset[i][k] for i in range(1,6)),
              "sequence_counts": dict(Counter(r["sequence"] for r in v).most_common())}
           for k,v in focused.items()}
    csv_write("case_scope_by_sequence.csv", [dict(sequence=s, **{k:c[k] for k in EXPECTED}) for s,c in sorted(by_sequence.items())])
    csv_write("case_scope_by_offset.csv", [dict(offset=i, **{k:by_offset[i][k] for k in EXPECTED}) for i in range(1,11)])
    with Path("case_scope.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(archive_sha256=sha, complete_windows=287, stable_windows=77,
                       first_interruptions=210, all_failure_counts=EXPECTED,
                       groups=one), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(one, ensure_ascii=False))


if __name__ == "__main__":
    main()
