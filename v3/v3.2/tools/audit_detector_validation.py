from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from counter_uas.evaluation.validation_set import audit_validation_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a human-labelled detector validation manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    audit = audit_validation_manifest(args.manifest, args.workspace_root, policy)
    payload = {
        "checkpoint": "3.2.4",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "policy": str(args.policy.resolve()),
        "policy_id": policy["policy_id"],
        "audit": audit.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["audit"], indent=2))
    return 0 if audit.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
