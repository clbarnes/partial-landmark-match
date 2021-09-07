from .constants import SRC_DIR
from collections import defaultdict
from typing import NamedTuple, Optional
import logging
import csv

import numpy as np

logger = logging.getLogger(__name__)


class Landmark(NamedTuple):
    id: int
    name: str
    location: np.ndarray
    group: Optional[str] = None

    @classmethod
    def from_line(cls, s, group=None, sep="\t"):
        s = s.strip()
        lmid, name, x, y, z, *_ = s.split("\t")
        return Landmark(int(lmid), name, np.array([float(x), float(y), float(z)]), group)


def parse_landmarks_txt(fpath):
    d = defaultdict(list)
    group = None
    with open(fpath) as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip()
            if not line.strip():
                continue
            if line.startswith("#"):
                group = line.lstrip("# ")
            elif line.startswith("\t"):
                d[group].append(Landmark.from_line(line))
            else:
                logger.warning("Skipping line %s, unexpected start: '%s'", line_idx, line)

    return d


def parse_entry_tsv(fpath):
    with open(fpath) as f:
        return [Landmark.from_line(ln) for ln in f]


def parse_entry_csv(fpath):
    out = []
    with open(fpath) as f:
        next(f)  # skip headers
        rdr = csv.reader(f)
        for row in rdr:
            lmid = int(row[1])
            name = "lm_" + row[1]
            loc = np.array([float(x) for x in row[3:6]])
            out.append(Landmark(lmid, name, loc))
    return out
