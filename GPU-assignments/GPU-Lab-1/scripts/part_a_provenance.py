"""Part A: capture nvidia-smi -q in full and record the identity fields."""
import argparse
import json
import os
import subprocess

import common


def capture_smi_q(index, outdir):
    """Full nvidia-smi -q dump, committed verbatim as the assignment requires."""
    out = subprocess.run(["nvidia-smi", "-q", "-i", str(index)],
                         capture_output=True, text=True, check=True)
    identity = common.gpu_identity(index)
    slug = identity["gpu_name"].replace("NVIDIA ", "").replace(" ", "_")
    path = os.path.join(outdir, f"nvidia-smi-q_{slug}_{identity['uuid'][-12:]}.txt")
    with open(path, "w") as fh:
        fh.write(out.stdout)
    return path, identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="GPU index to profile")
    args = ap.parse_args()

    common.require_cuda()
    path, identity = capture_smi_q(args.index, common.PROVENANCE)
    common.log_header("PART A provenance", identity)
    common.log("A", f"full nvidia-smi -q written to {os.path.relpath(path, common.ROOT)}",
               identity["uuid"])

    key, specs = common.load_specs(identity["gpu_name"])
    if specs is None:
        common.log("A", f"WARNING no vendor spec row for '{identity['gpu_name']}'. "
                        "Add one to scripts/gpu_specs.json", identity["uuid"])
    else:
        if not specs.get("verified"):
            common.log("A", f"WARNING vendor row '{key}' is marked verified=false. "
                            "Confirm against the cited source before submitting",
                       identity["uuid"])
        for field in ("architecture", "memory_type", "bandwidth_gb_s",
                      "tensor_core_generation", "tensor_core_precisions", "source"):
            common.log("A", f"{field}: {specs[field]}", identity["uuid"])

    record = {"identity": identity, "vendor_spec_key": key, "vendor_spec": specs,
              "smi_q_file": os.path.basename(path), "sid4": common.SID4, "seed": common.SEED}
    out = os.path.join(common.DATA, f"part_a_identity_{identity['uuid'][-12:]}.json")
    with open(out, "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {out}")

    for field in ("uuid", "driver_version", "cuda_version", "vram_total_mib", "power_limit_w"):
        common.log("A", f"{field}: {identity[field]}", identity["uuid"])


if __name__ == "__main__":
    main()
