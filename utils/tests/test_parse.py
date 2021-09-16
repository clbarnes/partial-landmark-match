from ..parse import parse_entry_csv, parse_entry_tsv, parse_landmarks_txt
from ..constants import SRC_DIR


def test_parse_entry_csv():
    landmarks = parse_entry_csv(SRC_DIR / "1099_lineage_entry_points-left.csv")
    assert len(landmarks)


def test_parse_entry_tsv():
    landmarks = parse_entry_tsv(SRC_DIR / "Seymour_lineage_entry_points.tsv")
    assert len(landmarks)


def test_parse_landmarks_txt():
    landmark_groups = parse_landmarks_txt(SRC_DIR / "Seymour_landmarks.txt")
    assert len(landmark_groups)
