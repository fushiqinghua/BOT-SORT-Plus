"""Replay two frozen M3OT validation clips with M2 and low-update factors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from m2_association import compose_m2


VAL = {"1_rgb_val_1-08": 600, "2_rgb_val_2-08": 600}
CONTROL = "B_early_moderate"
ARMS = {
    "m2_shadow": (False, False, False, False),
    "b_m2": (True, False, False, False),
    "b_bam_m2": (True, True, False, False),
    "b_low_r_m2": (True, False, True, False),
    "b_bam_low_r_m2": (True, True, True, False),
    "b_m2_full": (True, False, False, True),
    "b_bam_low_r_m2_full": (True, True, True, True),
}
FROZEN = ("native_config", "sequences", "detector_sha256", "rgb_encoder_sha256",
          "feature_hashes", "decoded_image_hashes", "detection_fingerprints",
          "encoder_contracts", "boxmot_source_hashes", "source_protocol_sha256",
          "implementation_sha256", "feature_coverage")
SOURCE_PINS = {
    "research_archive/botsort_research/association.py": "f96e391afe87d825058525a3f57dcadbd020d89949fd52221f28e62b24bdb6a4",
    "research_archive/botsort_research/noise.py": "48932df96768f6fde64b3111ff6e606a6938bad83adf44090a16e988f593a785",
    "research_archive/configs/research/round2/default.json": "e7aea9ca5ed11a0ada7a2865e285345e2c773466b66681c6214e747d7bbe24f4",
    "research_assets/m2_noise.json": "5c39407a1e865a9db72fcd3792a502424a85adec710a26bb11bfc6ac6adfe429",
}


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as opened:
        return hashlib.file_digest(opened, "sha256").hexdigest()


def store(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".writing")
    staging.write_text(json.dumps(item, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    staging.replace(path)


def load_references(project):
    first = project / "m3ot_first_batch_v1"
    archived = project / "m3ot_branch_replay_v1/tracking/rules"
    control = first / "tracking/run/higher"
    frozen = read(control / "frozen_inputs.json")
    protocol = read(control / "protocols" / (CONTROL + ".json"))
    if (frozen["status"] != "COMPLETE" or protocol["status"] != "COMPLETE"
            or protocol["sequences"] != VAL or not protocol["native_config"]["with_reid"]
            or read(first / "inputs/higher/ready.json")["status"] != "verified"):
        raise ValueError("The original B reference or frozen strong inputs are incomplete")
    if any(sha(archived / path) != value for path, value in SOURCE_PINS.items()):
        raise ValueError("An archived M2 measurement or association source changed")

    noise = read(archived / "research_assets/m2_noise.json")
    proof = read(archived / "receipts/fit_m2.json")
    previous = read(archived / "receipts/m2_nll_calibrated.json")
    training = noise["provenance"]
    if (proof["status"] != "FIT_ONLY" or set(proof["scope"]) != set(training["calibration_sequences"])
            or proof["used_validation_ground_truth"] or set(training["calibration_sequences"]) & VAL.keys()
            or set(training["evaluation_sequences"]) != VAL.keys()
            or training["detector_model_sha256"] != [protocol["detector_sha256"]]
            or previous["detector_sha256"] != protocol["detector_sha256"]
            or proof["fit_sha256"] != SOURCE_PINS["research_assets/m2_noise.json"]
            or noise["noise"]["mode"] != "calibrated" or noise["diagnostics"]["samples"] != 23061):
        raise ValueError("Unverified train-only M2 calibration")
    return first, archived, control, frozen, protocol, noise, previous


def run(project, out, arm):
    live, bam, low_r, full = ARMS[arm]
    first, archived, control, frozen, expected, noise, previous = load_references(project)
    repo = project.parent / "VT_Tiny_MOT_Baseline_v1/integrations/b_gmc_joint_r1/repo"
    sys.path[:0] = [str(path) for path in (repo, repo / "tools", project / "tracking_runtime",
                                           archived / "research_archive", out.parent / "low_update")]
    import run_botsort_plus as runner
    from botsort_plus import native_adapter
    from low_updates import compose_low_update

    base_prepare = runner.prepare
    config = dict(expected["config"], name=arm)
    cfg_path = out / "configs" / (arm + ".json")
    store(cfg_path, config)
    prior_config = read(archived / "research_archive/configs/research/round2/default.json")

    def configure(opts):
        cfg, paths, protocol = base_prepare(opts)
        if protocol["config"] != config:
            raise ValueError("B settings changed while composing M2")
        for key in FROZEN:
            if key in protocol and protocol[key] != expected[key]:
                raise ValueError(f"Frozen protocol differs at {key}")
        protocol["evaluation_context"] = "M3OT RGB two-drone validation on the fixed strong detector and RGB ReID"
        m2 = compose_m2(native_adapter.BoTSORTPlus, out, arm, paths, protocol, frozen, noise, prior_config,
                        live=live, full=full, low_r=low_r, store=store)
        protocol["m2_association"].update(
            archive_commit=previous["research_commit"],
            archive_protocol_sha256=previous["protocol_sha256"],
            archive_calibration_samples=noise["diagnostics"]["samples"],
            archived_sources=SOURCE_PINS,
            own_sources={name: sha(Path(__file__).with_name(name))
                         for name in ("replay.py", "m2_association.py")},
        )
        native_adapter.BoTSORTPlus = compose_low_update(
            m2, project, out, arm, paths, protocol, bam=bam, low_r=low_r
        )
        return cfg, paths, protocol

    runner.prepare = configure
    source = first / "inputs/higher"
    opts = runner.parser().parse_args([
        "--config", str(cfg_path), "--baseline-root", str(source),
        "--features-dir", str(source / "features"), "--images-root", str(project / "data/raw/M3OT"),
        "--runtime-root", str(project), "--output-dir", str(out), "--trace",
    ])
    runner.run(opts)
    protocol_path = out / "protocols" / (arm + ".json")
    protocol = read(protocol_path)
    if protocol["status"] != "COMPLETE":
        raise ValueError("Replay failed to complete")
    for key in FROZEN:
        if protocol[key] != expected[key]:
            raise ValueError(f"Completed tracker changed a frozen setting: {key}")
    receipts = {}
    for seq, frames in VAL.items():
        timing = read(out / "timing" / arm / (seq + ".json"))
        mot = out / "trackers" / arm / "data" / (seq + ".txt")
        events = out / "events" / arm / (seq + ".jsonl")
        m2_audit = out / "m2_events" / arm / (seq + ".json")
        low_audit = out / "low_update_events" / arm / (seq + ".json")
        if timing["frames"] != frames or timing["prediction_sha256"] != sha(mot):
            raise ValueError(f"Unexpected tracking output for {seq}")
        if read(low_audit)["native_sof_sha256"] != read(m2_audit)["native_sof_matrix_sha256"]:
            raise ValueError("The composed trackers observed different SOF streams")
        event_hash, mot_hash = sha(events), sha(mot)
        parity = None
        if arm == "m2_shadow":
            parity = {
                "tracks": mot_hash == expected["output_prediction_hashes"][seq],
                "native_events": event_hash == sha(control / "events" / CONTROL / events.name),
            }
            if not all(parity.values()):
                raise ValueError(f"M2 shadow changed B output or event bytes for {seq}")
        receipts[seq] = dict(frames=frames, rows=timing["output_rows"], sha256=mot_hash,
                             event_sha256=event_hash, m2_audit_sha256=sha(m2_audit),
                             low_audit_sha256=sha(low_audit), byte_parity=parity)
    store(out / "receipts" / (arm + ".json"), dict(
        arm=arm, status="COMPLETE", protocol_sha256=sha(protocol_path),
        detector_sha256=protocol["detector_sha256"], rgb_encoder_sha256=protocol["rgb_encoder_sha256"],
        calibration_sha256=SOURCE_PINS["research_assets/m2_noise.json"], sequences=receipts,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("arm", choices=ARMS)
    opts = parser.parse_args()
    run(opts.project, opts.out, opts.arm)
