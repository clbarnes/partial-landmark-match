from itertools import chain
from typing import Type

import numpy as np

from utils.transform import MatchAlgo, SimpleAlgo
from utils.parse import parse_entry_csv, parse_entry_tsv, parse_landmarks_txt
from utils.parse import SRC_DIR

# 1099 is src
# Seymour is tgt


def get_cps():
    src_landmarks = parse_landmarks_txt(SRC_DIR / "1099_landmarks.txt")
    tgt_landmarks = parse_landmarks_txt(SRC_DIR / "Seymour_landmarks.txt")
    src_by_fullname = {
        f"{lm.group}::{lm.name}": lm.location
        for lm in chain.from_iterable(src_landmarks.values())
    }
    tgt_by_fullname = {
        f"{lm.group}::{lm.name}": lm.location
        for lm in chain.from_iterable(tgt_landmarks.values())
    }

    src_cps = []
    tgt_cps = []
    for name, src_lm in src_by_fullname.items():
        try:
            tgt_lm = tgt_by_fullname[name]
        except KeyError:
            continue

        src_cps.append(src_lm)
        tgt_cps.append(tgt_lm)

    return np.array(src_cps), np.array(tgt_cps)


def get_other():
    src_lms = chain.from_iterable(
        parse_entry_csv(SRC_DIR / f"1099_lineage_entry_points-{side}.csv")
        for side in ["left", "right"]
    )

    tgt_lms = parse_entry_tsv(SRC_DIR / "Seymour_lineage_entry_points.tsv")
    return np.array([lm.location for lm in src_lms]), np.array(
        [lm.location for lm in tgt_lms]
    )


def print_eval(algo_class: Type[MatchAlgo]):
    src_cps, tgt_cps = get_cps()
    src_other, tgt_other = get_other()
    algo = algo_class(src_cps, tgt_cps, src_other, tgt_other)
    src_idx, tgt_idx, sum_dists, overlaps = algo.match()
    print(f"Cost for {algo_class.__name__}: {sum_dists}")


if __name__ == "__main__":
    print_eval(SimpleAlgo)
