from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    output = Path("audits/ub-sod-scene-v1/UB-SOD-scene-v1-audit.ipynb")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# UB-SOD scene-v1 split audit\n\n## tl;dr\n\n"
            "**Pass for controlled development.** The rebuilt split assigns every inferred visual component to exactly one partition and leaves zero cross-split pairs under the final leakage rule. External videos remain necessary for final field claims."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "This notebook audits the derived scene-grouped split, not the publisher split. Components are inferred from difference-hash candidates and direct resized-pixel similarity.\n\n"
            "### Key Assumptions\n\n"
            "Visual components are proxies because authoritative recording and airport IDs are unavailable. The looser sensitivity rule is expected to overmatch generic sky/runway images and is reviewed visually."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json\n\n"
            "audit_dir = Path('audits/ub-sod-scene-v1')\n"
            "split_dir = Path('splits/ub-sod-scene-v1')\n"
            "audit = json.loads((audit_dir / 'audit.json').read_text(encoding='utf-8'))\n"
            "manifest = json.loads((split_dir / 'split_manifest.json').read_text(encoding='utf-8'))"
        ),
        nbf.v4.new_markdown_cell("## Data\n\nEvery source image must occur in exactly one derived split."),
        nbf.v4.new_code_cell(
            "{\n 'decision': audit['decision'],\n 'source_images': audit['images'],\n 'assigned_images': audit['assigned_images'],\n"
            " 'missing_images': len(audit['missing_images']),\n 'unexpected_images': len(audit['unexpected_images'])\n}"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Split integrity and balance"),
        nbf.v4.new_code_cell("audit['split_stats']"),
        nbf.v4.new_markdown_cell("### Leakage checks"),
        nbf.v4.new_code_cell(
            "{\n 'groups_crossing_splits': len(audit['groups_crossing_splits']),\n"
            " 'final_rule_cross_split_pairs': len(audit['primary_cross_split_similarity_pairs']),\n"
            " 'looser_sensitivity_candidates': len(audit['sensitivity_cross_split_similarity_pairs']),\n"
            " 'largest_component': manifest['diagnostics']['largest_group']\n}"
        ),
        nbf.v4.new_markdown_cell(
            "The intentionally broader sensitivity screen is not a failure criterion. Its strongest remaining matches are visually reviewed below; generic sky/runway composition causes false matches once thresholds are loosened beyond the final rule."
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(audit_dir / 'sensitivity_pairs.jpg')))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. Freeze `scene-v1` for the first YOLO11s/YOLO11m comparison.\n"
            "2. Never substitute the publisher split in that comparison.\n"
            "3. Keep test untouched until model selection is complete.\n"
            "4. Treat UB-SOD results as development evidence; use external negative and distant-drone videos for field claims."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
