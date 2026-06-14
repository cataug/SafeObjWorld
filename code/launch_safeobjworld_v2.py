import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Job:
    name: str
    cmd: list
    log_path: Path


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_capture(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""


def query_gpus():
    out = run_capture([
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


def launch(job, gpu_id=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    ensure_dir(job.log_path.parent)
    f = open(job.log_path, "w", buffering=1)

    print(f"[launch] {job.name}")
    print(f"         gpu={gpu_id}")
    print(f"         log={job.log_path}")
    print("         cmd=" + " ".join(shlex.quote(str(x)) for x in job.cmd))

    proc = subprocess.Popen(
        job.cmd,
        stdout=f,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )
    return proc, f


def run_scheduler(
    jobs,
    min_free_mb=9000,
    max_jobs_per_gpu=2,
    poll_seconds=20,
    start_gap_seconds=10,
    cpu_parallel=2,
):
    pending = list(jobs)
    running = []

    print("=" * 100)
    print(f"Scheduler start. jobs={len(pending)}")
    print(f"min_free_mb={min_free_mb} max_jobs_per_gpu={max_jobs_per_gpu}")
    print("=" * 100)

    while pending or running:
        still = []
        for proc, log_f, job, gpu in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, log_f, job, gpu))
            else:
                log_f.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"[done] {job.name} gpu={gpu} status={status}")
                if ret != 0:
                    print(f"       log={job.log_path}")
        running = still

        gpus = query_gpus()

        if not gpus:
            running_cpu = sum(1 for _, _, _, gpu in running if gpu is None)
            while pending and running_cpu < cpu_parallel:
                job = pending.pop(0)
                proc, log_f = launch(job, None)
                running.append((proc, log_f, job, None))
                running_cpu += 1
                time.sleep(start_gap_seconds)
            time.sleep(poll_seconds)
            continue

        running_per_gpu = {g["index"]: 0 for g in gpus}
        for _, _, _, gpu in running:
            if gpu is not None and gpu in running_per_gpu:
                running_per_gpu[gpu] += 1

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
            proc, log_f = launch(job, gid)
            running.append((proc, log_f, job, gid))
            running_per_gpu[gid] = running_per_gpu.get(gid, 0) + 1
            launched_any = True
            time.sleep(start_gap_seconds)

        if not launched_any:
            msg = " | ".join([f"gpu{g['index']}:free={g['free_mb']}MB util={g['util']}%" for g in gpus])
            print(f"[wait] pending={len(pending)} running={len(running)} :: {msg}")

        time.sleep(poll_seconds)

    print("=" * 100)
    print("Scheduler finished.")
    print("=" * 100)


def maybe_add(jobs, name, cmd, log_dir, skip_path=None, force=False):
    if skip_path is not None and Path(skip_path).exists() and not force:
        print(f"[skip existing] {name} -> {skip_path}")
        return
    jobs.append(Job(name=name, cmd=cmd, log_path=Path(log_dir) / f"{name}.log"))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--project_root", type=str, default=str(Path.home() / "SafeObjWorld"))
    ap.add_argument("--dataset_root", type=str, default=str(Path.home() / "SafeObjWorld" / "data" / "DAVIS2017"))
    ap.add_argument("--run_root", type=str, default=str(Path.home() / "SafeObjWorld" / "runs_v2_full"))

    ap.add_argument("--image_size", type=int, default=192)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--prefix_len", type=int, default=3)
    ap.add_argument("--max_h", type=int, default=5)

    ap.add_argument("--run_h10", type=int, default=0)
    ap.add_argument("--run_optional_stress", type=int, default=0)

    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--min_free_mb", type=int, default=9000)
    ap.add_argument("--max_jobs_per_gpu", type=int, default=2)
    ap.add_argument("--poll_seconds", type=int, default=20)
    ap.add_argument("--start_gap_seconds", type=int, default=10)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--train_only", action="store_true")
    ap.add_argument("--eval_only", action="store_true")

    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    run_root = Path(args.run_root).expanduser().resolve()
    logs = ensure_dir(run_root / "logs")
    runner = project_root / "code" / "safeobjworld_v2.py"

    print("=" * 100)
    print("SafeObjWorld V2 launcher")
    print(json.dumps(vars(args), indent=2))
    print("=" * 100)

    base_common = [
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--image_size", str(args.image_size),
        "--batch_size", str(args.batch_size),
        "--max_train_samples", str(args.max_train_samples),
        "--max_val_samples", str(args.max_val_samples),
        "--num_workers", str(args.num_workers),
        "--seed", str(args.seed),
        "--skip_existing",
    ]
    if args.amp:
        base_common.append("--amp")

    max_h_values = [args.max_h]
    if args.run_h10:
        max_h_values.append(10)

    # --------------------------------------------------------
    # Phase 0: manifest
    # --------------------------------------------------------
    if not args.eval_only:
        cmd = [
            sys.executable, str(runner),
            "--mode", "make_manifest",
            "--dataset_root", str(dataset_root),
            "--run_root", str(run_root),
            "--prefix_len", str(args.prefix_len),
            "--max_h", str(args.max_h),
            "--image_size", str(args.image_size),
            "--seed", str(args.seed),
        ]
        subprocess.run(cmd, check=True)

    # --------------------------------------------------------
    # Phase 1: train learned methods
    # --------------------------------------------------------
    learned_methods = [
        # learned baseline
        "convlstm_base",
        "convlstm_stress",

        # proposed method + core ablations
        "safeobjworld_r_full",
        "safeobjworld_r_no_temporal",
        "safeobjworld_r_no_split",
        "safeobjworld_r_no_area",
        "safeobjworld_r_no_stress",
        "safeobjworld_r_det",
        "safeobjworld_r_image_only",
        "safeobjworld_r_mask_only",

        # architecture ablations from old idea
        "object_token_full",
        "object_token_det",
        "object_token_no_geom",
    ]

    train_jobs = []

    if not args.eval_only:
        for H in max_h_values:
            for method in learned_methods:
                ckpt = run_root / "models" / method / f"H{H}" / "best.pt"

                cmd = [
                    sys.executable, str(runner),
                    "--mode", "train",
                    "--method", method,
                    "--max_h", str(H),
                    "--epochs", str(args.epochs),
                    *base_common,
                ]

                maybe_add(
                    train_jobs,
                    name=f"train_{method}_H{H}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=ckpt,
                    force=args.force,
                )

        print(f"[phase 1] train jobs={len(train_jobs)}")
        run_scheduler(
            train_jobs,
            min_free_mb=args.min_free_mb,
            max_jobs_per_gpu=args.max_jobs_per_gpu,
            poll_seconds=args.poll_seconds,
            start_gap_seconds=args.start_gap_seconds,
        )

    if args.train_only:
        print("[train_only] stopping after training")
        return

    # --------------------------------------------------------
    # Phase 2: eval baselines + learned methods
    # --------------------------------------------------------
    baseline_methods = [
        "copy_last",
        "linear_centroid",
        "linear_centroid_scale",
        "flow_warp",
        "flow_linear_fallback",
    ]

    stresses = ["clean", "occlusion", "blur", "frame_drop"]
    if args.run_optional_stress:
        stresses += ["lowlight", "noise", "distractor"]

    eval_jobs = []

    for H in max_h_values:
        for stress in stresses:
            for method in baseline_methods:
                summary = run_root / "eval" / method / f"H{H}" / stress / "summary.csv"
                cmd = [
                    sys.executable, str(runner),
                    "--mode", "eval_baseline",
                    "--method", method,
                    "--stress", stress,
                    "--max_h", str(H),
                    *base_common,
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
                    sys.executable, str(runner),
                    "--mode", "eval_model",
                    "--method", method,
                    "--stress", stress,
                    "--max_h", str(H),
                    *base_common,
                ]
                maybe_add(
                    eval_jobs,
                    name=f"eval_{method}_H{H}_{stress}",
                    cmd=cmd,
                    log_dir=logs,
                    skip_path=summary,
                    force=args.force,
                )

    print(f"[phase 2] eval jobs={len(eval_jobs)}")
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
    cmd = [
        sys.executable, str(runner),
        "--mode", "aggregate",
        "--dataset_root", str(dataset_root),
        "--run_root", str(run_root),
        "--prefix_len", str(args.prefix_len),
        "--max_h", str(args.max_h),
        "--image_size", str(args.image_size),
        "--seed", str(args.seed),
    ]
    subprocess.run(cmd, check=True)

    print("=" * 100)
    print("DONE")
    print(f"RUN_ROOT={run_root}")
    print(f"RESULTS={run_root / 'RESULTS'}")
    print(f"LOGS={logs}")
    print("=" * 100)


if __name__ == "__main__":
    main()
