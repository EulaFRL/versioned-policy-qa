import xml.etree.ElementTree as ET
from datetime import date

from vpqa.data.ecfr import build_versions, parse_part, section_passages

PART = """<DIV5 N="9" TYPE="PART"><DIV8 N="9.1" TYPE="SECTION"><HEAD>§ 9.1 Benefits.</HEAD>
<XREF ID="1">Link to an amendment published at 81 FR 1.</XREF>
<P>(a) <I>General.</I> You may apply.</P>
<P>(1) Age 62 or older; or</P>
<P>(i) a sub-item under (1), not paragraph (i).</P>
<P>(b) The amount is $100.<CITA>[81 FR 1, Jan. 1, 2016]</CITA></P>
<SECAUTH>Authority: 42 U.S.C. 1.</SECAUTH>
</DIV8></DIV5>"""


def test_section_split_at_top_level_paragraphs():
    units = parse_part(PART.encode())
    assert list(units) == ["9.1::a", "9.1::b"]
    assert units["9.1::a"].startswith("§ 9.1 Benefits.\n(a) General.")
    assert "not paragraph (i)" in units["9.1::a"]
    assert "Link" not in units["9.1::a"] and "FR" not in units["9.1::b"] and "Authority" not in units["9.1::b"]


def test_long_paragraph_keyed_by_sub_paragraph_label():
    body = "".join(f"<P>({n}) {'word ' * 150}</P>" for n in (1, 2, 3))
    sec = ET.fromstring(f'<DIV8 N="9.2" TYPE="SECTION"><HEAD>§ 9.2 X.</HEAD><P>(a) Intro.</P>{body}</DIV8>')
    assert [k for k, _ in section_passages(sec)] == ["a", "a(1)", "a(2)", "a(3)"]


def test_versions_ignore_typesetting_and_close_on_change():
    snaps = [
        ("2017-01-01", {"9.1::a": "x with - y", "9.1::b": "$100"}),
        ("2018-01-01", {"9.1::a": "x with—y", "9.1::b": "$120"}),
        ("2019-01-01", {"9.1::a": "x with—y"}),
    ]
    vs = {v.pid: v for v in build_versions(snaps, 9, "9", "test")}
    assert set(vs) == {"9cfr9.1::a@2017-01-01", "9cfr9.1::b@2017-01-01", "9cfr9.1::b@2018-01-01"}
    assert vs["9cfr9.1::a@2017-01-01"].effective_to is None
    assert vs["9cfr9.1::b@2017-01-01"].effective_to == date(2018, 1, 1)
    assert vs["9cfr9.1::b@2018-01-01"].effective_to == date(2019, 1, 1)
