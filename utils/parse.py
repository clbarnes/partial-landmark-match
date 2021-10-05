from .constants import SRC_DIR
from collections import defaultdict
from typing import NamedTuple, Optional, DefaultDict, List
import logging
import csv

import numpy as np

logger = logging.getLogger(__name__)


class Landmark(NamedTuple):
    id: int
    location: np.ndarray
    name: Optional[str] = None
    group: Optional[str] = None

    @classmethod
    def from_line(cls, s, group=None, sep="\t"):
        s = s.strip()
        lmid, name, x, y, z, *_ = s.split(sep)
        return Landmark(
            int(lmid), np.array([float(x), float(y), float(z)]), name, group
        )

    @property
    def fullname(self) -> str:
        name = str(self.id) if self.name is None else self.name
        if self.group:
            return f"{self.group}::{name}"
        return name


def parse_landmarks_txt(fpath) -> DefaultDict[Optional[str], List[Landmark]]:
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
                d[group].append(Landmark.from_line(line, group=group))
            else:
                logger.warning(
                    "Skipping line %s, unexpected start: '%s'", line_idx, line
                )

    return d


def parse_landmarks_csv(fpath, group=None) -> List[Landmark]:
    out_lst = []
    with open(fpath) as f:
        headers = [h.strip() for h in next(f).split(",")]
        reader = csv.DictReader(f, fieldnames=headers)
        for row in reader:
            out_lst.append(
                Landmark(
                    int(row["treenode_id"]),
                    np.array([
                        float(row["x"]),
                        float(row["y"]),
                        float(row["z"]),
                    ]),
                    row["name"].strip(),
                    group,
                )
            )
    return out_lst


def parse_entry_tsv(fpath, group=None) -> List[Landmark]:
    with open(fpath) as f:
        return [Landmark.from_line(ln, group=group) for ln in f]


def parse_entry_csv(fpath, group=None) -> List[Landmark]:
    out = []
    with open(fpath) as f:
        next(f)  # skip headers
        rdr = csv.reader(f)
        for row in rdr:
            lmid = int(row[1])
            name = "lm_" + row[1].strip()
            loc = np.array([float(x) for x in row[3:6]])
            out.append(Landmark(lmid, loc, name, group=group))
    return out
