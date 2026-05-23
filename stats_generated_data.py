import argparse
from pathlib import Path


SPLIT_ALIASES = {
    "train": ["train"],
    "validation": ["validation", "val"],
    "test": ["test"],
}

TRAIN_REQUIRED_SUFFIXES = [
    ".down",
    ".feats",
    ".coords",
    ".vert_index",
    ".vert_gt",
    ".other_index",
    ".mini_line",
]
TEST_REQUIRED_SUFFIXES = [
    ".down",
    ".feats",
    ".coords",
    ".patch_index",
]
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _resolve_split_name(root, canonical_split):
    root = Path(root)
    for split_name in SPLIT_ALIASES[canonical_split]:
        xyz_dir = root / split_name / "xyz"
        if xyz_dir.exists():
            return split_name
    return None


def _replace_suffix(path, new_suffix):
    return Path(path).with_suffix(new_suffix)


def _is_non_empty(path):
    return path.exists() and path.stat().st_size > 0


def _fmt_ratio(count, total):
    if total == 0:
        return "0/0 (0.0%)"
    return f"{count}/{total} ({100.0 * count / total:.1f}%)"


def _collect_split_stats(noise_root, patch_root, split, sample_examples):
    actual_split = _resolve_split_name(noise_root, split)
    if actual_split is None:
        return {
            "split": split,
            "found": False,
        }

    xyz_files = sorted((Path(noise_root) / actual_split / "xyz").glob("*.xyz"))
    total = len(xyz_files)
    split_patch_root = Path(patch_root) / split

    stats = {
        "split": split,
        "found": True,
        "total": total,
        "all_outputs_present": 0,
        "patch_supervision_ready": 0,
        "line_supervision_ready": 0,
        "training_ready": 0,
        "missing_any_count": 0,
        "empty_down_count": 0,
        "empty_feats_count": 0,
        "empty_coords_count": 0,
        "empty_vert_index_count": 0,
        "empty_vert_gt_count": 0,
        "empty_other_index_count": 0,
        "empty_mini_line_count": 0,
        "empty_patch_index_count": 0,
        "missing_any_examples": [],
        "empty_vert_index_examples": [],
        "empty_mini_line_examples": [],
        "empty_patch_index_examples": [],
    }

    for xyz_file in xyz_files:
        save_root = split_patch_root / xyz_file.name

        down_path = _replace_suffix(save_root, ".down")
        feats_path = _replace_suffix(save_root, ".feats")
        coords_path = _replace_suffix(save_root, ".coords")

        if split == "test":
            patch_index_path = _replace_suffix(save_root, ".patch_index")
            required_paths = [down_path, feats_path, coords_path, patch_index_path]

            if all(path.exists() for path in required_paths):
                stats["all_outputs_present"] += 1
            else:
                stats["missing_any_count"] += 1
                if len(stats["missing_any_examples"]) < sample_examples:
                    stats["missing_any_examples"].append(xyz_file.name)

            if not _is_non_empty(down_path):
                stats["empty_down_count"] += 1
            if not _is_non_empty(feats_path):
                stats["empty_feats_count"] += 1
            if not _is_non_empty(coords_path):
                stats["empty_coords_count"] += 1
            if not _is_non_empty(patch_index_path):
                stats["empty_patch_index_count"] += 1
                if len(stats["empty_patch_index_examples"]) < sample_examples:
                    stats["empty_patch_index_examples"].append(xyz_file.name)
            continue

        vert_index_path = _replace_suffix(save_root, ".vert_index")
        vert_gt_path = _replace_suffix(save_root, ".vert_gt")
        other_index_path = _replace_suffix(save_root, ".other_index")
        mini_line_path = _replace_suffix(save_root, ".mini_line")
        required_paths = [
            down_path,
            feats_path,
            coords_path,
            vert_index_path,
            vert_gt_path,
            other_index_path,
            mini_line_path,
        ]

        if all(path.exists() for path in required_paths):
            stats["all_outputs_present"] += 1
        else:
            stats["missing_any_count"] += 1
            if len(stats["missing_any_examples"]) < sample_examples:
                stats["missing_any_examples"].append(xyz_file.name)

        if not _is_non_empty(down_path):
            stats["empty_down_count"] += 1
        if not _is_non_empty(feats_path):
            stats["empty_feats_count"] += 1
        if not _is_non_empty(coords_path):
            stats["empty_coords_count"] += 1
        if not _is_non_empty(vert_index_path):
            stats["empty_vert_index_count"] += 1
            if len(stats["empty_vert_index_examples"]) < sample_examples:
                stats["empty_vert_index_examples"].append(xyz_file.name)
        if not _is_non_empty(vert_gt_path):
            stats["empty_vert_gt_count"] += 1
        if not _is_non_empty(other_index_path):
            stats["empty_other_index_count"] += 1
        if not _is_non_empty(mini_line_path):
            stats["empty_mini_line_count"] += 1
            if len(stats["empty_mini_line_examples"]) < sample_examples:
                stats["empty_mini_line_examples"].append(xyz_file.name)

        patch_ready = (
            _is_non_empty(down_path)
            and _is_non_empty(feats_path)
            and _is_non_empty(coords_path)
            and _is_non_empty(vert_index_path)
            and _is_non_empty(vert_gt_path)
            and _is_non_empty(other_index_path)
        )
        line_ready = _is_non_empty(mini_line_path)
        training_ready = patch_ready and line_ready

        if patch_ready:
            stats["patch_supervision_ready"] += 1
        if line_ready:
            stats["line_supervision_ready"] += 1
        if training_ready:
            stats["training_ready"] += 1

    return stats


def _print_train_val_stats(stats):
    total = stats["total"]
    print(f"[{stats['split']}] total noisy point clouds: {total}")
    print(f"  all outputs present: {_fmt_ratio(stats['all_outputs_present'], total)}")
    print(f"  patch supervision ready: {_fmt_ratio(stats['patch_supervision_ready'], total)}")
    print(f"  line supervision ready: {_fmt_ratio(stats['line_supervision_ready'], total)}")
    print(f"  training ready: {_fmt_ratio(stats['training_ready'], total)}")
    print(f"  missing any required file: {_fmt_ratio(stats['missing_any_count'], total)}")
    print(f"  empty .vert_index: {_fmt_ratio(stats['empty_vert_index_count'], total)}")
    print(f"  empty .vert_gt: {_fmt_ratio(stats['empty_vert_gt_count'], total)}")
    print(f"  empty .other_index: {_fmt_ratio(stats['empty_other_index_count'], total)}")
    print(f"  empty .mini_line: {_fmt_ratio(stats['empty_mini_line_count'], total)}")
    if stats["missing_any_examples"]:
        print(f"  examples missing outputs: {', '.join(stats['missing_any_examples'])}")
    if stats["empty_vert_index_examples"]:
        print(f"  examples empty vert_index: {', '.join(stats['empty_vert_index_examples'])}")
    if stats["empty_mini_line_examples"]:
        print(f"  examples empty mini_line: {', '.join(stats['empty_mini_line_examples'])}")


def _print_test_stats(stats):
    total = stats["total"]
    print(f"[{stats['split']}] total noisy point clouds: {total}")
    print(f"  all outputs present: {_fmt_ratio(stats['all_outputs_present'], total)}")
    print(f"  empty .patch_index: {_fmt_ratio(stats['empty_patch_index_count'], total)}")
    print(f"  missing any required file: {_fmt_ratio(stats['missing_any_count'], total)}")
    if stats["missing_any_examples"]:
        print(f"  examples missing outputs: {', '.join(stats['missing_any_examples'])}")
    if stats["empty_patch_index_examples"]:
        print(f"  examples empty patch_index: {', '.join(stats['empty_patch_index_examples'])}")


def main():
    parser = argparse.ArgumentParser(description="Summarize generated PC2WF patch data quality.")
    parser.add_argument("--data_root", type=str, default=str(DEFAULT_DATA_ROOT), help="Dataset root that contains noise_sigma*/ and patches_*/.")
    parser.add_argument("--noise_root", type=str, default=None, help="Optional noisy point cloud root.")
    parser.add_argument("--patch_root", type=str, default=None, help="Optional generated patch root.")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch size used in the generated patch directory name.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Noise sigma used in the directory name.")
    parser.add_argument("--clip", type=float, default=0.01, help="Noise clip used in the directory name.")
    parser.add_argument("--show_examples", type=int, default=5, help="How many example file names to print for each category.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    noise_root = Path(args.noise_root) if args.noise_root else data_root / f"noise_sigma{args.sigma}clip{args.clip}"
    patch_root = Path(args.patch_root) if args.patch_root else data_root / f"patches_{args.patch_size}_noise_sigma{args.sigma}clip{args.clip}"

    print(f"noise_root={noise_root}")
    print(f"patch_root={patch_root}")
    print()

    for split in ("train", "validation", "test"):
        stats = _collect_split_stats(
            noise_root=noise_root,
            patch_root=patch_root,
            split=split,
            sample_examples=max(args.show_examples, 0),
        )
        if not stats["found"]:
            print(f"[{split}] noisy xyz directory not found under {noise_root}")
            print()
            continue

        if split == "test":
            _print_test_stats(stats)
        else:
            _print_train_val_stats(stats)
        print()


if __name__ == "__main__":
    main()
