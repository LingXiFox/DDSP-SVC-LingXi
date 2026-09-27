"""song_key() must collapse every slice of one song to one key, across layouts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_song_split_leakage import song_key  # noqa: E402


def test_stage1_builder_slices_carry_source_song_and_segment():
    # layout: <singer>_<song>/<singer>_<song>_<seg>.wav -> __s<idx> sub-slices
    assert song_key("36_一笑倾城_18__s0.wav") == "36_一笑倾城"
    assert song_key("36_云烟成雨_38__s1.wav") == "36_云烟成雨"
    # two segments of the same song share one key (this was the leak)
    assert song_key("9_大鱼_1.wav") == song_key("9_大鱼_12__s0.wav") == "9_大鱼"


def test_virtual_singer_slices_use_bare_s_suffix():
    # layout: r13_<track>_<seg>[s<idx>].wav
    assert song_key("r13_00_1.wav") == "r13_00"
    assert song_key("r13_00_15s0.wav") == "r13_00"
    assert song_key("r13_00_19s4.wav") == "r13_00"
    assert song_key("r13_11_3__s2.wav") == "r13_11"


def test_song_names_ending_in_digits_are_not_stripped_further():
    assert song_key("12_爱你一万年_7__s0.wav") == "12_爱你一万年"
    assert song_key("12_无段号歌名.wav") == "12_无段号歌名"
