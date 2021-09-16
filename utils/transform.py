from abc import abstractmethod
from typing import List, NamedTuple, Optional, Tuple

from molesq import Transformer
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


class TransformBuilder:
    def __init__(self):
        self.cp = []
        self.deformed_cp = []
        self.weights = None
        self._transformer = None

    @property
    def transformer(self):
        if self._transformer is None:
            self._transformer = Transformer(self.cp, self.deformed_cp, self.weights)
        return self.transformer

    def invalidate(self):
        self._transformer = None

    @property
    def ndim(self):
        if self.cp:
            return len(self.cp[0])
        return None

    def __len__(self):
        return len(self.cp)

    def append(self, cp, deformed_cp, weight=None):
        self.invalidate()
        ndim = self.ndim
        if ndim is not None and not (len(cp) == len(deformed_cp) == ndim):
            raise ValueError(f"Inconsistent dimensions, expected {ndim}")

        self.cp.append(cp)
        self.deformed_cp.append(deformed_cp)

        if len(self) == 0:
            if weight is not None:
                self.weights = [weight]
        elif self.weights is None:
            if weight is not None:
                raise ValueError("CPs must be all weighted or unweighted, not mixed")
        else:
            if weight is None:
                raise ValueError("CPs must be all weighted or unweighted, not mixed")
            else:
                self.weights.append(weight)
        return self

    def extend(self, cps, deformed_cps, weights=None):
        if weights is None or np.isscalar(weights):
            weights = [weights for _ in cps]
        for tup in zip(cps, deformed_cps, weights):
            self.append(*tup)
        return self

    def pop(self, idx=-1):
        self.invalidate()
        if self.weights is None:
            w = None
        else:
            w = self.weights.pop(idx)

        return self.cp.pop(idx), self.deformed_cp.pop(idx), w

    def clear(self):
        self.invalidate()
        self.cp.clear()
        self.deformed_cp.clear()
        self.weights = None

        return self


def clean_inputs(src_cps, tgt_cps, src_other, tgt_other):
    src_cps = np.asarray(src_cps)
    tgt_cps = np.asarray(tgt_cps)
    src_other = np.asarray(src_other)
    tgt_other = np.asarray(tgt_other)

    if src_cps.shape != tgt_cps.shape:
        raise ValueError("Control point arrays have different shapes")
    if src_cps.ndim != 2:
        raise ValueError("Control points must be 2D arrays")
    if src_other.ndim != 2 or tgt_other.ndim != 2:
        raise ValueError("Other points must be 2D arrays")

    ndim = len(src_cps[0])

    if src_other.shape[1] != ndim or tgt_other.shape[1] != ndim:
        raise ValueError(
            f"Other points must have same dimensionality as control points ({ndim})"
        )

    if len(src_other) > len(tgt_other):
        raise ValueError("More src_other points than tgt")

    return src_cps, tgt_cps, src_other, tgt_other


def match_evaluate(src_deformed: np.ndarray, tgt: np.ndarray, sqdist=True):
    """Non-unique. Cost is summed distance only"""
    metric = f"{'sq' if sqdist else ''}euclidean"
    dists = cdist(src_deformed, tgt, metric)
    src_idx = np.arange(len(src_deformed))
    tgt_idx = np.argmin(dists, axis=1)
    dist_sum = dists[src_idx, tgt_idx].sum()
    _, counts = np.unique(tgt_idx, return_counts=True)
    double_matches = np.sum(counts - 1)
    return src_idx, tgt_idx, dist_sum, double_matches


class MatchAlgo:
    def __init__(self, src_cps, tgt_cps, src_other, tgt_other):
        self.src_cps, self.tgt_cps, self.src_other, self.tgt_other = clean_inputs(
            src_cps, tgt_cps, src_other, tgt_other
        )
        self._transform: Optional[Transformer] = None

    @property
    def transform(self) -> Transformer:
        if self._transform is None:
            self._transform = self._make_transform()
        return self._transform

    @abstractmethod
    def _make_transform(self) -> Transformer:
        pass

    def match(self) -> Tuple[np.ndarray, np.ndarray, float, int]:
        transformed = self.transform.transform(self.src_other)
        return match_evaluate(transformed, self.tgt_other)


class SimpleAlgo(MatchAlgo):
    def _make_transform(self) -> Transformer:
        return Transformer(self.src_cps, self.tgt_cps)


# class BootstrapAlgo(MatchAlgo):
#     def _make_transform(self) -> Transformer:
#         # algorithm
#         # - sort unknown points by proximity to CPs (1/sum(dists)?)
#         # - iterate:
#         #   - pick a point
#         #   - transform it
#         #   - find its best 2 unpaired partners
#         #   - add this as a new CP, weighted by (sqdist_candidate1 / (sqdist_candidate1 + sqdist_candidate2))
#         tgt_loc_to_idx = {tuple(loc): idx for idx, loc in enumerate(self.tgt_other)}

#         builder = TransformBuilder().extend(self.src_cps, self.tgt_cps, 1)

#         sort_idxs = np.argsort(
#             cdist(self.src_cps, self.src_other, "euclidean").sum(axis=0)
#         )
#         src_other_sorted = list(self.src_other[sort_idxs])


class Matches(NamedTuple):
    neighbours: List
    dists: List

    @property
    def weights(self):
        d = np.array(self.dists)**2
        return list(1 - (d / d.sum()))

    def zip(self):
        return zip(self.neighbours, self.dists, self.weights)


class MatchRecommender:
    def __init__(self, points, labels=None, transformer=None) -> None:
        """`points` are not transformed; new points for querying are"""
        self.points = np.asarray(points)
        self.labels = self.tree.data if labels is None else np.asarray(labels)
        self.tree = KDTree(self.points)
        self.transformer = transformer

    def transform(self, points):
        if self.transformer is None:
            return points
        return self.transformer.transform(points)

    def match_points(self, points, n=5):
        dists, idxs = self.tree.query(self.transform(points), n)

        for dist_lst, idx_lst in zip(dists, idxs):
            yield Matches(self.labels[idx_lst], list(dist_lst))
