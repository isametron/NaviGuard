"""naviguard.cli — command-line entry point (`naviguard <subcommand>`)."""

import argparse
import os
import shutil
import sys


def _cmd_generate(args):
    from naviguard.data.generate import generate_dataset
    from naviguard.config import TELEMETRY_CSV

    df = generate_dataset(
        n_samples=args.n_samples,
        n_satellites=args.n_satellites,
        interval_s=args.interval_s,
        seed=args.seed,
        out_path=args.out or TELEMETRY_CSV,
    )
    print(f"[generate] {len(df)} rows written → {args.out or TELEMETRY_CSV}")
    print(f"[generate] Satellites: {args.n_satellites}  |  Interval: {args.interval_s}s")
    print("[generate] ✓ Complete")


def _cmd_preprocess(args):
    from naviguard.preprocessing.sequences import load_telemetry, fit_and_scale, build_sequences
    from naviguard.config import X_SEQ_PATH, Y_SEQ_PATH

    df = load_telemetry()
    df_scaled, _ = fit_and_scale(df)
    X, y = build_sequences(df_scaled, seq_len=args.seq_len, horizon=args.horizon)
    os.makedirs(os.path.dirname(X_SEQ_PATH), exist_ok=True)
    import numpy as np
    np.save(X_SEQ_PATH, X)
    np.save(Y_SEQ_PATH, y)
    print(f"[preprocess] Sequences  →  X:{X.shape}  y:{y.shape}")
    print(f"[preprocess] Saved: {X_SEQ_PATH}  |  {Y_SEQ_PATH}")
    print("[preprocess] ✓ Complete")


def _cmd_train(args):
    from naviguard.models.train import train
    train(seq_len=args.seq_len, horizon=args.horizon, epochs=args.epochs, batch_size=args.batch_size)


def _cmd_predict(args):
    from naviguard.inference.predict import evaluate_on_test, save_prediction_plot

    result = evaluate_on_test()
    print(f"[predict] Test samples  : {result['n_test_samples']}")
    for step, (mae, rmse) in enumerate(zip(result["mae_ns"], result["rmse_ns"]), start=1):
        marker = "  ← target ≤50ns" if step == 1 else ""
        print(f"[predict] Step {step:>2}: MAE={mae:.4f} ns  RMSE={rmse:.4f} ns{marker}")
    status = "✓ PASS" if result["pass_step1"] else "✗ EXCEEDS TARGET"
    print(f"[predict] Step 1 vs target ≤50ns: {status}")

    if args.save_plot:
        path = save_prediction_plot(result)
        print(f"[predict] Plot saved    : {path}")
    print("[predict] ✓ Complete")


def _cmd_serve(args):
    import uvicorn
    uvicorn.run("naviguard.api.main:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_clean(args):
    from naviguard.config import MODELS_DIR, OUTPUTS_DIR, X_SEQ_PATH, Y_SEQ_PATH

    targets = [MODELS_DIR, OUTPUTS_DIR, X_SEQ_PATH, Y_SEQ_PATH]
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
    _cmd_generate(args)
    _cmd_preprocess(args)
    _cmd_train(args)
    args.save_plot = True
    _cmd_predict(args)
    print("\n[pipeline] ALL STEPS COMPLETE")
    print("[pipeline] Launch the API: uvicorn naviguard.api.main:app --reload")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="naviguard", description="NaviGuard — satellite clock intelligence CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic NavIC/GNSS telemetry")
    p_gen.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    p_gen.add_argument("--n-satellites", type=int, default=1, dest="n_satellites")
    p_gen.add_argument("--interval-s", type=int, default=900, dest="interval_s")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--out", type=str, default=None)
    p_gen.set_defaults(func=_cmd_generate)

    p_pre = sub.add_parser("preprocess", help="Scale + build sliding-window sequences")
    p_pre.add_argument("--seq-len", type=int, default=20, dest="seq_len")
    p_pre.add_argument("--horizon", type=int, default=6)
    p_pre.set_defaults(func=_cmd_preprocess)

    p_train = sub.add_parser("train", help="Train the attention-LSTM model")
    p_train.add_argument("--seq-len", type=int, default=20, dest="seq_len")
    p_train.add_argument("--horizon", type=int, default=6)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p_train.set_defaults(func=_cmd_train)

    p_pred = sub.add_parser("predict", help="Evaluate on the held-out test split")
    p_pred.add_argument("--save-plot", action="store_true", dest="save_plot")
    p_pred.set_defaults(func=_cmd_predict)

    p_serve = sub.add_parser("serve", help="Run the FastAPI service")
    p_serve.add_argument("--host", type=str, default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=_cmd_serve)

    p_clean = sub.add_parser("clean", help="Remove generated models/outputs/sequence artifacts")
    p_clean.set_defaults(func=_cmd_clean)

    p_pipe = sub.add_parser("pipeline", help="Run generate -> preprocess -> train -> predict in sequence")
    p_pipe.add_argument("--n-samples", type=int, default=200, dest="n_samples")
    p_pipe.add_argument("--n-satellites", type=int, default=1, dest="n_satellites")
    p_pipe.add_argument("--interval-s", type=int, default=900, dest="interval_s")
    p_pipe.add_argument("--seed", type=int, default=42)
    p_pipe.add_argument("--out", type=str, default=None)
    p_pipe.add_argument("--seq-len", type=int, default=20, dest="seq_len")
    p_pipe.add_argument("--horizon", type=int, default=6)
    p_pipe.add_argument("--epochs", type=int, default=50)
    p_pipe.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    p_pipe.set_defaults(func=_cmd_pipeline)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
