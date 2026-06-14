
import argparse
import os
import sys
import time
import json
import shlex
import subprocess
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Job:
    name: str
    cmd: list
    kind: str = "gpu"
    cwd: str = None
    env: dict = field(default_factory=dict)
    log_path: Path = None


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_cmd_capture(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def query_gpus():
    out = run_cmd_capture([
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    gpus = []
    if not out:
        return gpus

    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "free_mb": int(parts[1]),
                "total_mb": int(parts[2]),
                "util": int(parts[3]),
            })
        except Exception:
            pass
    return gpus


def launch_job(job, gpu_id=None):
    env = os.environ.copy()
    env.update(job.env)

    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"

    ensure_dir(job.log_path.parent)
    f = open(job.log_path, "w", buffering=1)

    print(f"[launch] {job.name}")
    print(f"         gpu={gpu_id}")
    print(f"         log={job.log_path}")
    print(f"         cmd={' '.join(shlex.quote(str(x)) for x in job.cmd)}")

    proc = subprocess.Popen(
        job.cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        cwd=job.cwd,
        env=env,
        text=True,
    )
    return proc, f


def run_scheduler(
    jobs,
    min_free_mb=8500,
    max_jobs_per_gpu=3,
    poll_seconds=20,
    start_gap_seconds=12,
    cpu_parallel=4,
):
    pending = list(jobs)
    running = []

    print("=" * 100)
    print(f"Scheduler started. Jobs={len(pending)}")
    print(f"min_free_mb={min_free_mb} max_jobs_per_gpu={max_jobs_per_gpu}")
    print("=" * 100)

    while pending or running:
        # Check completed
        still = []
        for item in running:
            proc, log_f, job, gpu = item
            ret = proc.poll()
            if ret is None:
                still.append(item)
            else:
                log_f.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[done] {job.name} gpu={gpu} status={status}")
                if ret != 0:
                    print(f"       log: {job.log_path}")
        running = still

        # CPU-only fallback
        gpus = query_gpus()
        if not gpus:
            running_cpu = sum(1 for _, _, _, gpu in running if gpu is None)
            while pending and running_cpu < cpu_parallel:
                job = pending.pop(0)
                proc, log_f = launch_job(job, gpu_id=None)
                running.append((proc, log_f, job, None))
                running_cpu += 1
                time.sleep(1)
            time.sleep(poll_seconds)
            continue

        running_per_gpu = {g["index"]: 0 for g in gpus}
        for _, _, _, gpu in running:
            if gpu is not None and gpu in running_per_gpu:
                running_per_gpu[gpu] += 1

        # Prefer GPUs with more free memory
        gpus = sorted(gpus, key=lambda x: x["free_mb"], reverse=True)

        launched_any = False

        for g in gpus:
            if not pending:
                break

            gid = g["index"]
            if running_per_gpu.get(gid, 0) >= max_jobs_per_gpu:
                continue

            if g["free_mb"] < min_free_mb:
                continue

            job = pending.pop(0)
            proc, log_f = launch_job(job, gpu_id=gid)
            running.append((proc, log_f, job, gid))
            running_per_gpu[gid] = running_per_gpu.get(gid, 0) + 1
            launched_any = True
            time.sleep(start_gap_seconds)

        if not launched_any:
            gpu_msg = " | ".join([f"gpu{g['index']}:free={g['free_mb']}MB util={g['util']}%" for g in gpus])
            print(f"[wait] pending={len(pending)} running={len(running)} :: {gpu_msg}")

        time.sleep(poll_seconds)

    print("=" * 100)
    print("Scheduler finished.")
    print("=" * 100)


def maybe_add(jobs, name, cmd, log_dir, skip_path=None, force=False):
    if skip_path is not None and Path(skip_path).exists() and not force:
        print(f"[skip job exists] {name} -> {skip_path}")
        return
    jobs.append(Job(name=name, cmd=cmd, log_path=Path(log_dir) / f"{name}.log"))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default=str(Path.home() / "SafeObjWorld"))
    ap.add_argument("--dataset_root", type=str, default=str(Path.home() / "SafeObjWorld" / "data" / "DAVIS2017"))
    ap.add_argument("--run_root", type=str, default=str(Path.home() / "SafeObjWorld" / "runs_musthave"))

    ap.add_argument("--image_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--prefix_len", type=int, default=3)
    ap.add_argument("--max_h", type=int, default=5)
    ap.add_argument("--run_h10", type=int, default=0)

    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--min_free_mb", type=int, default=8500)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=3)
    ap.add_argument("--poll_seconds", type=int, default=20)
    ap.add_argument("--start_gap_seconds", type=int, default=12)

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    code_script = project_root / "code" / "safeobjworld_run.py"

    logs = ensure_dir(run_root / "logs")

    print("=" * 100)
    print("SafeObjWorld Must-have launcher")
    print(json.dumps(vars(args), indent=2))
    print("=" * 100)

    # --------------------------------------------------------
    # Phase 0: manifest
    # --------------------------------------------------------
    manifest_cmd = [
        sys.executable, str(code_script),
        "--mode", "make_manifest",
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--max_h", str(args.max_h),
        "--image_size", str(args.image_size),
        "--seed", str(args.seed),
    ]
    print("[phase 0] manifest")
    subprocess.run(manifest_cmd, check=True)

    # --------------------------------------------------------
    # Phase 1: train learned methods
    # --------------------------------------------------------
    max_h_values = [args.max_h]
    if args.run_h10:
        max_h_values.append(10)

    train_methods = [
        "convlstm",
        "safeobj_full",
        "safeobj_det",
        "safeobj_no_geom",
        "safeobj_no_stress",
    ]

    train_jobs = []
    for H in max_h_values:
        for method in train_methods:
            ckpt = run_root / "models" / method / f"H{H}" / "best.pt"

            stress_train = 0 if method == "safeobj_no_stress" else 1
            temporal_weight = 0.0 if method == "convlstm" else 0.05

            cmd = [
                sys.executable, str(code_script),
                "--mode", "train",
                "--dataset_root", str(dataset_root),
                "--run_root", str(run_root),
                "--method", method,
                "--prefix_len", str(args.prefix_len),
                "--max_h", str(H),
                "--image_size", str(args.image_size),
                "--batch_size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--max_train_samples", str(args.max_train_samples),
                "--max_val_samples", str(args.max_val_samples),
                "--num_workers", str(args.num_workers),
                "--stress_train", str(stress_train),
                "--temporal_weight", str(temporal_weight),
                "--seed", str(args.seed),
                "--skip_existing",
            ]
            if args.amp:
                cmd.append("--amp")

            maybe_add(
                train_jobs,
                name=f"train_{method}_H{H}",
                cmd=cmd,
                log_dir=logs,
                skip_path=ckpt,
                force=args.force,
            )

    print(f"[phase 1] train jobs: {len(train_jobs)}")
    run_scheduler(
        train_jobs,
        min_free_mb=args.min_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        poll_seconds=args.poll_seconds,
        start_gap_seconds=args.start_gap_seconds,
    )

    # --------------------------------------------------------
    # Phase 2: eval baselines + learned models
    # --------------------------------------------------------
    stresses = ["clean", "occlusion", "blur", "frame_drop"]

    baseline_methods = [
        "copy_last",
        "linear_centroid",
        "flow_warp",
    ]

    learned_methods = train_methods

    eval_jobs = []

    for H in max_h_values:
        for stress in stresses:
            for method in baseline_methods:
                summary = run_root / "eval" / method / f"H{H}" / stress / "summary.csv"
                cmd = [
                    sys.executable, str(code_script),
                    "--mode", "eval_baseline",
                    "--dataset_root", str(dataset_root),
                    "--run_root", str(run_root),
                    "--method", method,
                    "--stress", stress,
                    "--prefix_len", str(args.prefix_len),
                    "--max_h", str(H),
                    "--image_size", str(args.image_size),
                    "--batch_size", str(args.batch_size),
                    "--max_val_samples", str(args.max_val_samples),
                    "--num_workers", str(args.num_workers),
                    "--seed", str(args.seed),
                    "--skip_existing",
                ]
                maybe_add(
                    eval_jobs,
                    name=f"eval_{method}_H{H}_{stress}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=summary,
                    force=args.force,
                )

            for method in learned_methods:
                summary = run_root / "eval" / method / f"H{H}" / stress / "summary.csv"
                cmd = [
                    sys.executable, str(code_script),
                    "--mode", "eval_model",
                    "--dataset_root", str(dataset_root),
                    "--run_root", str(run_root),
                    "--method", method,
                    "--stress", stress,
                    "--prefix_len", str(args.prefix_len),
                    "--max_h", str(H),
                    "--image_size", str(args.image_size),
                    "--batch_size", str(args.batch_size),
                    "--max_val_samples", str(args.max_val_samples),
                    "--num_workers", str(args.num_workers),
                    "--eval_k", "5",
                    "--seed", str(args.seed),
                    "--skip_existing",
                ]
                if args.amp:
                    cmd.append("--amp")

                maybe_add(
                    eval_jobs,
                    name=f"eval_{method}_H{H}_{stress}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=summary,
                    force=args.force,
                )

    print(f"[phase 2] eval jobs: {len(eval_jobs)}")
    run_scheduler(
        eval_jobs,
        min_free_mb=args.min_free_mb,
        max_jobs_per_gpu=args.max_jobs_per_gpu,
        poll_seconds=args.poll_seconds,
        start_gap_seconds=args.start_gap_seconds,
    )

    # --------------------------------------------------------
    # Phase 3: aggregate
    # --------------------------------------------------------
    print("[phase 3] aggregate")
    agg_cmd = [
        sys.executable, str(code_script),
        "--mode", "aggregate",
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--max_h", str(args.max_h),
        "--image_size", str(args.image_size),
        "--seed", str(args.seed),
    ]
    subprocess.run(agg_cmd, check=True)

    print("=" * 100)
    print("DONE")
    print(f"Results root: {run_root}")
    print(f"Main table:    {run_root / 'RESULTS' / 'table_main_reliability.csv'}")
    print(f"All summaries: {run_root / 'RESULTS' / 'all_summaries.csv'}")
    print(f"All rows:      {run_root / 'RESULTS' / 'all_rows.csv'}")
    print(f"Logs:          {logs}")
    print("=" * 100)


if __name__ == "__main__":
    main()
