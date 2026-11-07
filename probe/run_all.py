"""Reproduce the probe results: download data, evaluate, and draw figures.

  python -m probe.run_all                  full pipeline
  python -m probe.run_all --skip-download  reuse data/activations.npz and data/runs.json
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    data = ["--data-dir", args.data_dir]
    both = data + ["--results-dir", args.results_dir]
    steps = [] if args.skip_download else [("probe.extract", data)]
    steps += [("probe.evaluate", both), ("probe.figures", both)]
    for module, options in steps:
        print(f"=== {module}", flush=True)
        subprocess.run([sys.executable, "-m", module, *options], check=True)


if __name__ == "__main__":
    main()
