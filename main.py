#!/usr/bin/env python3
from itertools import chain
from typing import Type, Iterable

import logging

import numpy as np
from molesq import Transformer
import pandas as pd
from matplotlib import pyplot as plt

from utils.transform import MatchAlgo, SimpleAlgo, MatchRecommender
from utils.parse import Landmark, parse_entry_csv, parse_entry_tsv, parse_landmarks_txt
from utils.parse import SRC_DIR

# 1099 is src
# Seymour is tgt

logger = logging.getLogger(__name__)

DIMS = ["x", "y", "z"]

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

    tgt_lms = parse_entry_tsv(SRC_DIR / "Seymour_lineage_entry_points.tsv")
    logger.info("got %s src, %s tgt points", len(src_lms), len(tgt_lms))
    return landmarks_df(src_lms), landmarks_df(tgt_lms)


def print_eval(algo_class: Type[MatchAlgo]):
    src_cps, tgt_cps = get_cps()
    src_other, tgt_other = get_other()
    algo = algo_class(src_cps, tgt_cps, src_other[DIMS], tgt_other[DIMS])
    src_idx, tgt_idx, sum_dists, overlaps = algo.match()
    print(f"Cost for {algo_class.__name__}: {sum_dists}")


def recommend():
    src_cps, tgt_cps = get_cps()

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


def plot_points():
    src_cps, tgt_cps = get_cps()

    src_other, tgt_other = get_other()
    transformer = Transformer(src_cps, tgt_cps)
    transformed_src = transformer.transform(src_other[DIMS])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*tgt_cps.T, label="seymour control points")
    ax.scatter(*tgt_other[DIMS].to_numpy().T, label="seymour entry points")
    # ax.scatter(*transformed_src.T, label="1099 entry points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # print_eval(SimpleAlgo)
    recommend()
    # plot_points()
