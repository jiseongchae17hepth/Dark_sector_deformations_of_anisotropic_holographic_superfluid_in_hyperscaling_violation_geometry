import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_z_policy import write_alpha_z_policy_outputs


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "alpha_z_policy"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    hsv_summary = load_json(output_root / "hsv_solver_forks" / "hsv_solver_forks_summary.json")
    result = write_alpha_z_policy_outputs(Path(args.output_dir), hsv_result=hsv_summary)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
