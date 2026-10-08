from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    output = Path("audits/ub-sod/UB-SOD-audit.ipynb")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# UB-SOD training-readiness audit\n\n"
            "## tl;dr\n\n"
            "**Conditional use only.** The 7,118 images and 18,042 YOLO boxes are structurally valid, "
            "but the publisher-provided image-level splits are not reliable independent evaluation splits. "
            "Direct pixel checks find strong train-to-holdout scene similarity in hundreds of images. "
            "Use the images for training only after rebuilding splits by recording/scene group; retain a separate independent test set."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "The intended use is fine-tuning and evaluating a ground-camera UAV-versus-bird detector. "
            "The audit validates every image and YOLO row, profiles class and box-size distributions, hashes files, "
            "screens cross-split perceptual similarity, and inspects fixed visual samples.\n\n"
            "### Key Assumptions\n\n"
            "- Class 0 is UAV and class 1 is Bird, as declared by `data.yaml`.\n"
            "- A true generalization test must separate recording sequences/scenes, not merely filenames.\n"
            "- Perceptual hashes are candidate generators; direct resized-pixel correlation is used as confirmation, not semantic proof."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport csv, json\n\n"
            "audit_dir = Path('audits/ub-sod')\n"
            "audit = json.loads((audit_dir / 'audit.json').read_text(encoding='utf-8'))\n"
            "with (audit_dir / 'near_pairs.csv').open(encoding='utf-8') as handle:\n"
            "    near_pairs = list(csv.DictReader(handle))\n"
            "len(near_pairs)"
        ),
        nbf.v4.new_markdown_cell("## Data\n\nThe following values come from a complete scan of the extracted release."),
        nbf.v4.new_code_cell(
            "{\n"
            "    'images': audit['images'],\n"
            "    'split_images': audit['split_images'],\n"
            "    'instances': audit['instances'],\n"
            "    'class_instances': audit['class_instances'],\n"
            "    'empty_label_images': audit['empty_label_images'],\n"
            "    'validation_errors': len(audit['validation_errors']),\n"
            "}"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n### Annotation integrity passes"),
        nbf.v4.new_code_cell(
            "size = audit['object_size']\n"
            "{\n"
            "    'COCO-small boxes': size['coco_small_lt_1024'],\n"
            "    'COCO-small rate': f\"{size['coco_small_rate']:.1%}\",\n"
            "    'median box area px': round(size['area_px_median'], 1),\n"
            "    'boxes with minimum side < 4 px': size['min_dimension_lt_4px'],\n"
            "    'boxes with minimum side < 8 px': size['min_dimension_lt_8px'],\n"
            "}"
        ),
        nbf.v4.new_markdown_cell(
            "### Publisher splits show material scene leakage\n\n"
            "There are no byte-identical images across splits, but scene-level independence fails the stronger screen."
        ),
        nbf.v4.new_code_cell(
            "holdout = audit['split_images']['val'] + audit['split_images']['test']\n"
            "correlations = [float(row['pixel_correlation']) for row in near_pairs]\n"
            "{\n"
            "    'holdout images': holdout,\n"
            "    'dHash-near holdout-to-train pairs': len(near_pairs),\n"
            "    'pixel correlation >= 0.95': sum(value >= 0.95 for value in correlations),\n"
            "    'share of holdout >= 0.95': f\"{sum(value >= 0.95 for value in correlations) / holdout:.1%}\",\n"
            "    'pixel correlation >= 0.98': sum(value >= 0.98 for value in correlations),\n"
            "    'share of holdout >= 0.98': f\"{sum(value >= 0.98 for value in correlations) / holdout:.1%}\",\n"
            "}"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(audit_dir / 'near_duplicate_candidates.jpg')))"
        ),
        nbf.v4.new_markdown_cell(
            "### Small-target coverage is useful but demanding\n\n"
            "The dataset strongly represents small targets, including 816 boxes with a minimum side below 8 pixels. "
            "These cases are useful for training but may be below recoverable detail after resizing to 640 pixels."
        ),
        nbf.v4.new_code_cell("display(Image(filename=str(audit_dir / 'small_targets.jpg')))"),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "1. **Do not use the supplied validation/test scores as the primary proof of generalization.**\n"
            "2. **Do not discard the dataset.** Its annotations and small-target coverage are useful for training.\n"
            "3. Reconstruct splits by recording or visually derived scene group before model selection.\n"
            "4. Add independent negative videos and an external locked test set. There are no empty-label images in this release.\n"
            "5. Start YOLO11s only after the split repair; compare YOLO11m with the identical repaired split."
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    print(output)


if __name__ == "__main__":
    main()
