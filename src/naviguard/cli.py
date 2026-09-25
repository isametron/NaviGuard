"""naviguard.cli — command-line entry point (`naviguard <subcommand>`)."""

import argparse
import os
import shutil
import sys
import time

from naviguard import config as cfg
from naviguard.data import generate as gen


def _cmd_generate(args):
    df = gen.generate_dataset(
        n_samples=args.n_samples,
        n_satellites=args.n_satellites,
        interval_s=args.interval_s,
        seed=args.seed,
        out_path=args.out or cfg.TELEMETRY_CSV,
        anomaly_count=args.anomaly_count,
    )
    print(f"[generate] {len(df)} rows written → {args.out or cfg.TELEMETRY_CSV}")
    print(f"[generate] Satellites: {args.n_satellites}  |  Interval: {args.interval_s}s"
          f"  |  Injected anomalies/sat: {args.anomaly_count}")
    print("[generate] ✓ Complete")


def _cmd_preprocess(args):
    from naviguard.preprocessing.sequences import (
        build_sequences, fit_scaler, load_telemetry, save_sequences,
    )

    df = load_telemetry(args.telemetry or cfg.TELEMETRY_CSV)
    scaler = fit_scaler(df)
    seqs = build_sequences(df, scaler, seq_len=args.seq_len, horizon=args.horizon)
    save_sequences(seqs)
    print(f"[preprocess] {len(df)} rows  |  scaler fit on training rows only → {cfg.SCALER_PATH}")
    print(f"[preprocess] Sequences  →  X:{seqs.X.shape}  y:{seqs.y.shape}")
    print(f"[preprocess] Windows: train={seqs.count('train')}  val={seqs.count('val')}  test={seqs.count('test')}")
    print(f"[preprocess] Saved: {cfg.SEQUENCES_PATH}")
    print("[preprocess] ✓ Complete")


def _cmd_train(args):
    from naviguard.models.train import train

    hparams = cfg.ModelHParams(dropout=args.dropout, learning_rate=args.lr)
    train(seq_len=args.seq_len, horizon=args.horizon, epochs=args.epochs,
          batch_size=args.batch_size, patience=args.patience, hparams=hparams)


def _cmd_predict(args):
    from naviguard.inference.predict import (
        detect_anomalies, evaluate_split, save_prediction_plot,
    )

    result = evaluate_split(split=args.split)
    print(f"[predict] Split: {args.split}  |  samples: {result['n_test_samples']}")
    base = result["baselines"]
    print(f"[predict] {'step':>4} {'model':>10} {'persist':>10} {'linear':>10} {'ridge':>10}   (MAE, ns)")
    for i, mae in enumerate(result["mae_ns"]):
        print(f"[predict] {i + 1:>4} {mae:>10.3f} {base['persistence']['mae_ns'][i]:>10.3f} "
              f"{base['linear_extrapolation']['mae_ns'][i]:>10.3f} {base['ridge']['mae_ns'][i]:>10.3f}")
    status = "✓ PASS" if result["pass_step1"] else "✗ EXCEEDS TARGET"
    print(f"[predict] Step 1 vs target ≤{cfg.MAE_TARGET_NS:g}ns: {status}")
    skill = result["skill_vs_persistence"]
    if skill is not None:
        print(f"[predict] Skill vs persistence (step 1): {skill:+.1%}")

    if args.split == "test":
        det = detect_anomalies()
        print(f"[predict] Anomaly scan: {det['n_flagged']}/{det['n_scored']} flagged "
              f"(|z| > {det['z_threshold']:g}, max |z| = {det['max_abs_z']:.1f})  →  {det['severity']}")

    if args.save_plot:
        path = save_prediction_plot(result)
        print(f"[predict] Plot saved    : {path}")
    print("[predict] ✓ Complete")


def _cmd_fetch(args):
    from datetime import date, timedelta

    from naviguard.data.broadcast import fetch_range, to_telemetry

    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=2)
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days - 1)
    records, failed = fetch_range(start, end, keep_raw=args.keep_raw, force=args.force)
    tel = to_telemetry(records, min_records=args.min_records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tel.to_csv(args.out, index=False)
    print(f"[fetch] {start} → {end}: {len(records)} IRNSS records, {len(failed)} day(s) failed")
    for sat, g in tel.groupby("satellite_id"):
        print(f"[fetch]   I{sat:02d}: {len(g)} samples")
    print(f"[fetch] Saved: {args.out}")
    print("[fetch] ✓ Complete")


def _cmd_benchmark(args):
    from naviguard.benchmark.models import ALL_MODELS
    from naviguard.benchmark.run import run_benchmark, run_pooled_benchmark

    if args.pooled:
        pooled_models = tuple(args.models.split(",")) if args.models else ("ridge", "arima_p10", "lstm", "gru", "attn_lstm")
        run_pooled_benchmark(
            csv_path=args.csv, out_dir=args.out_dir, models=pooled_models, seq_len=args.seq_len,
            horizon=args.horizon, n_folds=args.folds, seeds=args.seeds,
            min_windows=args.min_windows, epochs=args.epochs, resume=args.resume,
        )
        print(f"[benchmark] Pooled rows merged into {args.out_dir}/results.csv (models suffixed _pooled)")
        return
    models = tuple(args.models.split(",")) if args.models else ALL_MODELS
    run_benchmark(
        csv_path=args.csv, out_dir=args.out_dir,
        satellites=[int(s) for s in args.satellites.split(",")] if args.satellites else None,
        models=models, seq_len=args.seq_len, horizon=args.horizon, n_folds=args.folds,
        seeds=args.seeds, min_windows=args.min_windows, epochs=args.epochs, resume=args.resume,
    )
    print(f"[benchmark] Results in {args.out_dir}  (results.csv, summary.md, summary.tex, dm_tests.csv)")
    print("[benchmark] ✓ Complete")


def _cmd_anomaly_eval(args):
    from naviguard.benchmark.anomaly_eval import run_anomaly_eval

    run_anomaly_eval(args.csv, args.out_dir, seq_len=args.seq_len, n_events=args.events, trials=args.trials)
    print(f"[anomaly] Summary: {args.out_dir}/summary.md")
    print("[anomaly] ✓ Complete")


def _cmd_train_navic(args):
    from naviguard.inference.navic import train_navic

    meta = train_navic(csv_path=args.csv, seq_len=args.seq_len, horizon=args.horizon, epochs=args.epochs,
                       seed=args.seed)
    print(f"[train-navic] Serve it with:  NAVIGUARD_PROFILE=navic naviguard serve   (model: {meta['hparams']['kind']})")
    print("[train-navic] ✓ Complete")


def _cmd_report(args):
    from naviguard.benchmark.figures import build_report

    bench = {}
    for spec in args.bench:
        label, _, path = spec.partition("=")
        bench[label] = path
    made = build_report(bench, args.anomaly_dir, args.csv, args.out_dir)
    for p in made:
        print(f"[report] {p}")
    print("[report] ✓ Complete")


def _cmd_serve(args):
    import uvicorn
    uvicorn.run("naviguard.api.main:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_clean(args):
    targets = [cfg.MODELS_DIR, cfg.OUTPUTS_DIR, cfg.SEQUENCES_PATH]
    for path in targets:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"[clean] Removed dir : {path}")
        elif os.path.isfile(path):
            os.remove(path)
            print(f"[clean] Removed file: {path}")
        else:
            print(f"[clean] Not found   : {path}")
    print("[clean] ✓ Complete")


def _cmd_pipeline(args):
    start = time.time()
    for name, fn in (("generate", _cmd_generate), ("preprocess", _cmd_preprocess),
                     ("train", _cmd_train), ("predict", _cmd_predict)):
        print(f"\n{'=' * 60}\n  STEP: {name}\n{'=' * 60}")
        t0 = time.time()
        if name == "predict":
            args.save_plot = True
            args.split = "test"
        fn(args)
        print(f"[pipeline] {name} completed in {time.time() - t0:.1f}s")
    print(f"\n[pipeline] ALL STEPS COMPLETE  |  total {time.time() - start:.1f}s")
    print("[pipeline] Launch the API: naviguard serve")


def _add_generate_args(p):
    p.add_argument("--n-samples", type=int, default=gen.DEFAULT_N_SAMPLES, dest="n_samples")
    p.add_argument("--n-satellites", type=int, default=1, dest="n_satellites")
    p.add_argument("--interval-s", type=int, default=gen.DEFAULT_INTERVAL_S, dest="interval_s")
    p.add_argument("--seed", type=int, default=gen.DEFAULT_SEED)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--anomaly-count", type=int, default=0, dest="anomaly_count",
                   help="labelled anomalies to inject into each satellite's test region")


def _add_window_args(p):
    p.add_argument("--seq-len", type=int, default=cfg.SEQ_LEN, dest="seq_len")
    p.add_argument("--horizon", type=int, default=cfg.HORIZON)


def _add_train_args(p):
    p.add_argument("--epochs", type=int, default=cfg.DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=cfg.DEFAULT_BATCH_SIZE, dest="batch_size")
    p.add_argument("--patience", type=int, default=cfg.DEFAULT_PATIENCE)
    p.add_argument("--dropout", type=float, default=cfg.ModelHParams.dropout)
    p.add_argument("--lr", type=float, default=cfg.ModelHParams.learning_rate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="naviguard", description="NaviGuard — satellite clock intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic NavIC/GNSS telemetry")
    _add_generate_args(p_gen)
    p_gen.set_defaults(func=_cmd_generate)

    p_pre = sub.add_parser("preprocess", help="Fit scaler, split chronologically, build windows")
    _add_window_args(p_pre)
    p_pre.add_argument("--telemetry", type=str, default=None)
    p_pre.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser("train", help="Train the attention-LSTM model")
    _add_window_args(p_train)
    _add_train_args(p_train)
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Evaluate vs baselines + anomaly scan on a held-out split")
    p_pred.add_argument("--split", choices=["val", "test"], default="test")
    p_pred.add_argument("--save-plot", action="store_true", dest="save_plot")
    p_pred.set_defaults(func=_cmd_predict)

    p_fetch = sub.add_parser("fetch", help="Download real NavIC broadcast clock data (BRDM, per satellite)")
    p_fetch.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (default: end - days + 1)")
    p_fetch.add_argument("--end", type=str, default=None, help="YYYY-MM-DD (default: 2 days ago)")
    p_fetch.add_argument("--days", type=int, default=7)
    p_fetch.add_argument("--min-records", type=int, default=48, dest="min_records",
                         help="drop satellites with fewer records (sparsely tracked)")
    p_fetch.add_argument("--out", type=str, default=os.path.join(cfg.DATA_DIR, "navic_telemetry.csv"))
    p_fetch.add_argument("--keep-raw", action="store_true", dest="keep_raw")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_bench = sub.add_parser("benchmark", help="Rolling-origin benchmark of forecasters on real NavIC series")
    p_bench.add_argument("--csv", type=str, default=os.path.join(cfg.DATA_DIR, "navic_telemetry.csv"))
    p_bench.add_argument("--out-dir", type=str, default=os.path.join(cfg.OUTPUTS_DIR, "benchmark"), dest="out_dir")
    p_bench.add_argument("--satellites", type=str, default=None, help="comma-separated PRNs (default: all)")
    p_bench.add_argument("--models", type=str, default=None, help="comma-separated (default: all)")
    p_bench.add_argument("--seq-len", type=int, default=cfg.SEQ_LEN, dest="seq_len")
    p_bench.add_argument("--horizon", type=int, default=cfg.HORIZON)
    p_bench.add_argument("--folds", type=int, default=3)
    p_bench.add_argument("--seeds", type=int, default=3)
    p_bench.add_argument("--min-windows", type=int, default=200, dest="min_windows")
    p_bench.add_argument("--epochs", type=int, default=60)
    p_bench.add_argument("--resume", action="store_true", help="skip satellites already finished in --out-dir")
    p_bench.add_argument("--pooled", action="store_true",
                         help="train one model on all satellites (merged into an existing --out-dir)")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_anom = sub.add_parser("anomaly-eval", help="Evaluate anomaly detectors on real series (injected + real events)")
    p_anom.add_argument("--csv", type=str, default=os.path.join(cfg.DATA_DIR, "navic_telemetry.csv"))
    p_anom.add_argument("--out-dir", type=str, default=os.path.join(cfg.OUTPUTS_DIR, "anomaly"), dest="out_dir")
    p_anom.add_argument("--seq-len", type=int, default=cfg.SEQ_LEN, dest="seq_len")
    p_anom.add_argument("--events", type=int, default=8, help="faults injected per trial")
    p_anom.add_argument("--trials", type=int, default=5)
    p_anom.set_defaults(func=_cmd_anomaly_eval)

    p_tn = sub.add_parser("train-navic", help="Train + select a model on real NavIC data (navic profile)")
    p_tn.add_argument("--csv", type=str, default=cfg.NAVIC_TELEMETRY_CSV)
    p_tn.add_argument("--seq-len", type=int, default=cfg.SEQ_LEN, dest="seq_len")
    p_tn.add_argument("--horizon", type=int, default=cfg.HORIZON)
    p_tn.add_argument("--epochs", type=int, default=40)
    p_tn.add_argument("--seed", type=int, default=0)
    p_tn.set_defaults(func=_cmd_train_navic)

    p_rep = sub.add_parser("report", help="Build paper figures/LaTeX tables from benchmark + anomaly outputs")
    p_rep.add_argument("--bench", action="append", default=[],
                       help="LABEL=DIR of a benchmark output (repeatable), e.g. 'horizon 6=outputs/benchmark_h6'")
    p_rep.add_argument("--anomaly-dir", type=str, default=os.path.join(cfg.OUTPUTS_DIR, "anomaly"), dest="anomaly_dir")
    p_rep.add_argument("--csv", type=str, default=os.path.join(cfg.DATA_DIR, "navic_telemetry.csv"))
    p_rep.add_argument("--out-dir", type=str, default=os.path.join(cfg.OUTPUTS_DIR, "figures"), dest="out_dir")
    p_rep.set_defaults(func=_cmd_report)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_clean = sub.add_parser("clean", help="Remove generated models/outputs/sequence artifacts")
    p_clean.set_defaults(func=_cmd_clean)

    p_pipe = sub.add_parser("pipeline", help="Run generate -> preprocess -> train -> predict in sequence")
    _add_generate_args(p_pipe)
    _add_window_args(p_pipe)
    _add_train_args(p_pipe)
    p_pipe.add_argument("--telemetry", type=str, default=None)
    p_pipe.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        # Expected operator errors (missing telemetry/artifacts, bad sizes): message, not traceback.
        print(f"[naviguard] ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
