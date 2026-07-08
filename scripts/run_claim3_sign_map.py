import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis_workflows import run_claim3_sign_map


def parse_float_list(raw: str):
    if raw is None:
        return None
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "d5_claim3"))
    parser.add_argument("--delta2", default=None, help="Comma-separated subset, e.g. 0.10,0.15")
    parser.add_argument("--alpha-dm", default=None, help="Comma-separated subset")
    parser.add_argument("--mu-x", default=None, help="Comma-separated subset")
    args = parser.parse_args()

    result = run_claim3_sign_map(
        Path(args.output_dir),
        delta2_values=parse_float_list(args.delta2),
        alpha_dm_values=parse_float_list(args.alpha_dm),
        mu_x_values=parse_float_list(args.mu_x),
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
