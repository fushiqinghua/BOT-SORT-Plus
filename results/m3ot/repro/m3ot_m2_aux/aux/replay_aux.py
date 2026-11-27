"""Run fixed auxiliary association experiments on the M3OT validation clips."""
import argparse
from pathlib import Path
import sys

from aux_association import compose_aux


CONTROL = "b_bam_low_r"
ARMS = {
    "m2_aux_shadow": "shadow",
    "b_bam_low_r_m2_aux": "soft",
    "b_bam_low_r_m2_gate": "gate",
    "b_bam_low_r_m2_high_nll": "high_nll",
    "b_bam_low_r_m2_low_nll": "low_nll",
    "b_bam_low_r_no_high_reid": "no_high_reid",
}


def run(project, out, arm):
    sys.path.insert(0, str(out.parent / "m2"))
    from m2_association import with_changes
    from replay import FROZEN, SOURCE_PINS, VAL, load_references, read, sha, store

    first, archived, _, frozen, b_protocol, noise, archived_m2 = load_references(project)
    reference_folder = out.parent / "low_update"
    control_path = reference_folder / "protocols" / (CONTROL + ".json")
    reference = read(control_path)
    if (reference["status"] != "COMPLETE" or
            any(reference[k] != b_protocol[k] for k in FROZEN) or
            not reference["low_updates"]["bam"] or not reference["low_updates"]["measurement_r"]):
        raise ValueError("Frozen B, BAM and learned-low-R control does not match its inputs")
    source_config = read(archived / "research_archive/configs/research/round2/default.json")
    repo = project.parent / "VT_Tiny_MOT_Baseline_v1/integrations/b_gmc_joint_r1/repo"
    sys.path[:0] = [str(path) for path in (repo, repo / "tools", reference_folder,
                                           project / "tracking_runtime", archived / "research_archive")]
    import run_botsort_plus as runner
    from botsort_plus import native_adapter
    from low_updates import compose_low_update

    base_prepare = runner.prepare
    config = dict(reference["config"], name=arm)
    cfg_path = out / "configs" / (arm + ".json")
    store(cfg_path, config)

    def configure(opts):
        cfg, paths, protocol = base_prepare(opts)
        if (protocol["config"] != config or
                any(protocol[k] != reference[k] for k in FROZEN if k in protocol)):
            raise ValueError("B parameters or frozen inputs changed before replay")
        aux = compose_aux(native_adapter.BoTSORTPlus, paths, protocol, out, arm, ARMS[arm],
                          noise, source_config, with_changes=with_changes, store=store)
        protocol["evaluation_context"] = (
            "M3OT strong detector plus fixed RGB ReID caches, two RGB validation sequences"
        )
        protocol["m2_aux"].update(
            source_pins=SOURCE_PINS, archive_commit=archived_m2["research_commit"],
            source_b_bam_low_r_protocol_sha256=sha(control_path),
            implementation_sha256={name: sha(Path(__file__).with_name(name))
                                   for name in ("aux_association.py", "replay_aux.py")},
        )
        native_adapter.BoTSORTPlus = compose_low_update(
            aux, project, out, arm, paths, protocol, bam=True, low_r=True
        )
        if protocol["low_updates"] != reference["low_updates"]:
            raise ValueError("BAM or low-R implementation/contract changed")
        return cfg, paths, protocol

    runner.prepare = configure
    assets = first / "inputs/higher"
    args = runner.parser().parse_args([
        "--config", str(cfg_path), "--baseline-root", str(assets),
        "--features-dir", str(assets / "features"), "--images-root", str(project / "data/raw/M3OT"),
        "--runtime-root", str(project), "--output-dir", str(out), "--trace",
    ])
    runner.run(args)
    protocol_path = out / "protocols" / (arm + ".json")
    protocol = read(protocol_path)
    if (protocol["status"] != "COMPLETE" or
            any(protocol[k] != reference[k] for k in (*FROZEN, "low_updates"))):
        raise ValueError("Replay incomplete or changed frozen components")
    results = {}
    for seq, frames in VAL.items():
        timing = read(out / "timing" / arm / (seq + ".json"))
        track = out / "trackers" / arm / "data" / (seq + ".txt")
        events = out / "events" / arm / (seq + ".jsonl")
        audit_path = out / "aux_events" / arm / (seq + ".json")
        decision_path = out / "aux_decisions" / arm / (seq + ".jsonl.gz")
        low_path = out / "low_update_events" / arm / (seq + ".json")
        low_audit = read(low_path)
        audit = read(audit_path)
        expected_sof = frozen["action_counts"]["B_early_moderate"][seq]["native_matrix_sha256"]
        if (timing["frames"] != frames or sha(track) != timing["prediction_sha256"]
                or low_audit["native_sof_sha256"] != expected_sof
                or audit["stages"]["low"]["applied_pairs"] != low_audit["r_updates"]
                or low_audit["r_updates"] != low_audit["accepted_low"]
                or low_audit["both_updates"] != low_audit["bam_updates"]):
            raise ValueError("M2 replay changed the fixed camera or low-update scope")
        parity = None
        if ARMS[arm] == "shadow":
            old_tracks = reference_folder / "trackers" / CONTROL / "data" / track.name
            old_events = reference_folder / "events" / CONTROL / events.name
            old_low = reference_folder / "low_update_events" / CONTROL / low_path.name
            parity = dict(mot=sha(old_tracks) == sha(track) == reference["output_prediction_hashes"][seq],
                          events=sha(old_events) == sha(events),
                          bam_and_low_r_events=sha(old_low) == sha(low_path))
            if not all(parity.values()):
                raise ValueError(f"Auxiliary shadow changed reference bytes: {seq}")
        results[seq] = dict(rows=timing["output_rows"], frames=frames, sha256=sha(track),
                            native_event_sha256=sha(events), aux_audit_sha256=sha(audit_path),
                            aux_decision_sha256=sha(decision_path), low_update_sha256=sha(low_path),
                            byte_parity=parity)
    store(out / "receipts" / (arm + ".json"), dict(
        arm=arm, status="COMPLETE", sequences=results, protocol_sha256=sha(protocol_path),
        calibration_sha256=SOURCE_PINS["research_assets/m2_noise.json"],
        detector_sha256=protocol["detector_sha256"], rgb_encoder_sha256=protocol["rgb_encoder_sha256"],
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("arm", choices=ARMS)
    options = parser.parse_args()
    run(options.project, options.out, options.arm)
