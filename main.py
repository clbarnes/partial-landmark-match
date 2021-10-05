#!/usr/bin/env python3
from itertools import chain
from typing import Type, Iterable, Dict, Tuple, Optional

import logging

import numpy as np
from molesq import Transformer
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from utils.transform import MatchAlgo, SimpleAlgo, MatchRecommender
from utils.parse import Landmark, parse_entry_csv, parse_entry_tsv, parse_landmarks_txt, parse_landmarks_csv
from utils.parse import SRC_DIR

# 1099 is src
# Seymour is tgt

logger = logging.getLogger(__name__)

DIMS = ["x", "y", "z"]
DEFAULT_ANTERIOR_ONLY = False


def flat_cps_dicts(anterior_only):
    src_landmarks = parse_landmarks_txt(SRC_DIR / "1099_landmarks.txt")
    tgt_landmarks = parse_landmarks_txt(SRC_DIR / "Seymour_landmarks.txt")

    if anterior_only:
        src_landmarks = {
            k: v for k, v in src_landmarks.items()
            if "brain" in k
        }
        tgt_landmarks = {
            k: v for k, v in tgt_landmarks.items()
            if "brain" in k
        }

    src_by_fullname = {
        f"{lm.group}::{lm.name}": lm.location
        for lm in chain.from_iterable(src_landmarks.values())
    }
    src_by_fullname.update({
        lm.name: lm.location
        for lm in parse_landmarks_csv(SRC_DIR / "1099_commissure_antennaLobe.csv")
    })
    tgt_by_fullname = {
        f"{lm.group}::{lm.name}": lm.location
        for lm in chain.from_iterable(tgt_landmarks.values())
    }
    tgt_by_fullname.update({
        lm.name: lm.location
        for lm in parse_landmarks_csv(SRC_DIR / "Seymour_commissure_antennaLobe.csv")
    })
    return src_by_fullname, tgt_by_fullname


def get_cps(anterior_only):
    src_by_fullname, tgt_by_fullname = flat_cps_dicts(anterior_only)

    src_cps = []
    tgt_cps = []
    for name, src_lm in src_by_fullname.items():
        try:
            tgt_lm = tgt_by_fullname[name]
        except KeyError:
            continue

        src_cps.append(src_lm)
        tgt_cps.append(tgt_lm)

    logger.info("got %s src, %s tgt control points", len(src_cps), len(tgt_cps))

    return np.array(src_cps), np.array(tgt_cps)


def landmarks_df(lms: Iterable[Landmark]):
    return pd.DataFrame(
        data=[list(lm.location) + [lm.fullname] for lm in lms],
        columns=DIMS + ["name"],
    )


def get_other():
    src_lms = list(chain.from_iterable(
        parse_entry_csv(SRC_DIR / f"1099_lineage_entry_points-{side}.csv", side)
        for side in ["left", "right"]
    ))

    tgt_lms = []
    for side in ["left", "right"]:
        tgt_lms.extend(parse_entry_tsv(SRC_DIR / f"Seymour_lineage_entry_points_{side}.tsv", group=side))
    logger.info("got %s src, %s tgt points", len(src_lms), len(tgt_lms))
    return landmarks_df(src_lms), landmarks_df(tgt_lms)


def print_eval(algo_class: Type[MatchAlgo], anterior_only=DEFAULT_ANTERIOR_ONLY):
    src_cps, tgt_cps = get_cps(DEFAULT_ANTERIOR_ONLY)
    src_other, tgt_other = get_other()
    algo = algo_class(src_cps, tgt_cps, src_other[DIMS], tgt_other[DIMS])
    src_idx, tgt_idx, sum_dists, overlaps = algo.match()
    print(f"Cost for {algo_class.__name__}: {sum_dists}")


def recommend(anterior_only=DEFAULT_ANTERIOR_ONLY):
    src_cps, tgt_cps = get_cps(anterior_only)

    src_other, tgt_other = get_other()
    transformer = Transformer(src_cps, tgt_cps)
    transformed_src = transformer.transform(src_other[DIMS])
    recommender = MatchRecommender(tgt_other[DIMS], tgt_other["name"])
    for src_name, match in zip(
        src_other["name"], recommender.match_points(transformed_src)
    ):
        print(src_name)
        for neighbour, dist, w in match.zip():
            print(f"\t{dist:.1f}\t{w:.2f}\t{neighbour}")


def plot_points(anterior_only=DEFAULT_ANTERIOR_ONLY):
    src_cps, tgt_cps = get_cps(anterior_only)

    src_other, tgt_other = get_other()
    transformer = Transformer(src_cps, tgt_cps)
    transformed_src = transformer.transform(src_other[DIMS])

    scatter_points({
        "seymour control points": tgt_cps,
        "seymour entry points": tgt_other[DIMS].to_numpy(),
        "transformed 1099 entry points": transformed_src,
    })
    plt.show()


def split_lr(landmarks: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for k, v in landmarks.items():
        if "left" in k:
            if "right" in k:
                logger.warning("ambiguous left/right split for '%s'", k)
            else:
                left.append(v)
        elif "right" in k:
            right.append(v)
    return np.array(left), np.array(right)


def scatter_points(point_groups: Dict[str, np.ndarray], ax: Optional[Axes] = None):
    """points is Nx3"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    for label, points in point_groups.items():
        ax.scatter(*np.asarray(points).T, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    return ax


def plot_cps(anterior_only=DEFAULT_ANTERIOR_ONLY):
    src, tgt = flat_cps_dicts(anterior_only)
    fig = plt.figure()

    src_l, src_r = split_lr(src)
    tgt_l, tgt_r = split_lr(tgt)

    ax_src = fig.add_subplot(projection='3d')
    # ax_tgt = fig.add_subplot(projection='3d')

    scatter_points({"src L": src_l, "src R": src_r}, ax_src)
    # scatter_points({"tgt L": tgt_l, "tgt R": tgt_r}, ax_tgt)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # print_eval(SimpleAlgo)
    recommend(False)
    #plot_points(False)
    # plot_cps()
