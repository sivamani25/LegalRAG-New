import pytest
from legalrag.ingest.chunker import (
    SentenceChunker, PatternChunker, Chunker, make_chunk_id, Chunk
)

def test_make_chunk_id():
    # Tests normalization
    assert make_chunk_id("Acts", "IPC 1860", 42) == "acts/ipc_1860/42"
    assert make_chunk_id("Legal QA", "some/path", 0) == "legal_qa/some_path/0"
    assert make_chunk_id("$$", "!", 1) == "unknown/unknown/1"

def test_sentence_chunker_basic():
    c = SentenceChunker(max_chars=200, min_chars=10)
    text = "Hello world. This is a test. Short! Another sentence."
    chunks = c.chunk("doc", text, "cat", "jur")
    assert len(chunks) > 0
    assert "Hello world. This is a test." in chunks[0].text
    assert chunks[0].section is None

def test_sentence_chunker_min_chars_accumulation():
    c = SentenceChunker(max_chars=100, min_chars=40)
    text = "A. B. C. D. E. F. G. H. I. J."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 1
    assert chunks[0].text == "A. B. C. D. E. F. G. H. I. J."

def test_sentence_chunker_max_chars_window():
    c = SentenceChunker(max_chars=30, min_chars=10)
    text = "This is a long sentence that is over thirty characters. This is another one."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 2
    assert chunks[0].text == "This is a long sentence that is over thirty characters."
    assert chunks[1].text == " This is another one."

def test_sentence_chunker_empty():
    c = SentenceChunker()
    assert c.chunk("doc", "") == []

def test_sentence_chunker_offsets():
    c = SentenceChunker(max_chars=2, min_chars=0)
    text = "A. B. C."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 3
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == 2
    assert chunks[1].start_offset == 2
    assert chunks[1].end_offset == 5
    assert chunks[2].start_offset == 5
    assert chunks[2].end_offset == 8

def test_pattern_chunker_generic():
    c = PatternChunker()
    text = "Intro.\nSection 1\nBody one.\n1.1\nSub.\nSection 2\nBody two."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 4
    # Preamble
    assert chunks[0].section is None
    assert chunks[0].text == "Intro."
    # Section 1
    assert chunks[1].section == "Section 1"
    assert chunks[1].heading == "Section 1"
    assert chunks[1].text == "Body one."
    assert chunks[1].parent is None
    # 1.1
    assert chunks[2].section == "1.1"
    assert chunks[2].parent == "Section 1"
    assert chunks[2].text == "Sub."
    # Section 2
    assert chunks[3].section == "Section 2"
    assert chunks[3].text == "Body two."

def test_pattern_chunker_acts_preset():
    delimiters = [
        {"pattern": r"^[ \t]*(?:Chapter|Part)\s+[IVXLCDM\d]+", "level": 1},
        {"pattern": r"^[ \t]*(?:Section)\s+\d+[A-Z]?", "level": 2},
        {"pattern": r"^[ \t]*\(\d+\)", "level": 3},
        {"pattern": r"^[ \t]*\([a-z]\)", "level": 4}
    ]
    c = PatternChunker(delimiters)
    text = "Chapter I\nPreliminary\nSection 1\nTitle.\n(1)\nSub one.\n(a)\nClause."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 4
    assert chunks[0].section == "Chapter I"
    assert chunks[0].text == "Preliminary"
    assert chunks[0].hierarchy_path == ["Chapter I"]
    assert chunks[1].section == "Section 1"
    assert chunks[1].text == "Title."
    assert chunks[1].hierarchy_path == ["Chapter I", "Section 1"]
    assert chunks[2].section == "1"
    assert chunks[2].parent == "Section 1"
    assert chunks[2].hierarchy_path == ["Chapter I", "Section 1", "1"]
    assert chunks[3].section == "a"
    assert chunks[3].parent == "1"
    assert chunks[3].hierarchy_path == ["Chapter I", "Section 1", "1", "a"]

def test_pattern_chunker_constitution():
    delimiters = [
        {"pattern": r"^[ \t]*(?:Part|Schedule)\s+[IVX]+", "level": 1},
        {"pattern": r"^[ \t]*(?:Article)\s+\d+", "level": 2}
    ]
    c = PatternChunker(delimiters)
    text = "Part III\nRights\nArticle 21\nLife.\nSchedule I\nStates."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 3
    assert chunks[0].section == "Part III"
    assert chunks[1].section == "Article 21"
    assert chunks[1].parent == "Part III"
    assert chunks[2].section == "Schedule I"
    assert chunks[2].parent is None

def test_pattern_chunker_empty():
    c = PatternChunker()
    assert c.chunk("doc", "") == []

def test_pattern_chunker_no_delimiters():
    c = PatternChunker()
    chunks = c.chunk("doc", "Just a block of text.")
    assert len(chunks) == 1
    assert chunks[0].section is None

def test_pattern_chunker_heading_capture():
    c = PatternChunker()
    text = "Section 5\nThis is the heading\nBody starts here."
    chunks = c.chunk("doc", text)
    # The heading is the first line, body starts after
    assert chunks[0].heading == "Section 5"
    assert chunks[0].text == "This is the heading\nBody starts here."

def test_chunker_facade_sentence():
    c = Chunker("sentence")
    assert isinstance(c._impl, SentenceChunker)
    chunks = c.chunk("doc", "A. B.")
    # With lookahead regex, length is preserved and it flushes at max_chars.
    # Text is "A. B." len 5, smaller than max_chars 1000. It produces 1 chunk!
    assert len(chunks) == 1

def test_chunker_facade_pattern():
    presets = {"test_preset": {"delimiters": [{"pattern": r"\d+\.", "level": 1}]}}
    c = Chunker("pattern", preset_name="test_preset", presets=presets)
    assert isinstance(c._impl, PatternChunker)
    chunks = c.chunk("doc", "1. One\n2. Two")
    assert len(chunks) == 2
    assert chunks[0].section == "1"

def test_chunker_facade_invalid():
    with pytest.raises(ValueError):
        Chunker("unknown")

def test_chunk_to_dict():
    chunk = Chunk("id", "doc", "path", "cat", "jur", "txt", "orig", "sum", "sec", "par", "head", ["p"], [{"cites": 1}], 0, 10, True)
    d = chunk.to_dict()
    assert d["chunk_id"] == "id"
    assert d["cites"] == [{"cites": 1}]
    assert d["hierarchy_path"] == ["p"]

def test_pattern_chunker_cuad_contracts():
    delimiters = [
        {"pattern": r"^[ \t]*[Aa]rticle\s+[IVXLCDM]+\b", "level": 1},
        {"pattern": r"^[ \t]*[A-Z][A-Z\s]{4,}[A-Z]\b", "level": 1},
        {"pattern": r"^[ \t]*(?:Section|§)\s+\d+", "level": 2},
        {"pattern": r"^[ \t]*\d+\.\d+(?:\.\d+)*", "level": 3},
        {"pattern": r"^[ \t]*\([a-z]\)", "level": 4}
    ]
    c = PatternChunker(delimiters)
    text = "CONFIDENTIALITY\n(a)\nSecret.\nARTICLE IV\nSection 1\nRule."
    chunks = c.chunk("doc", text)
    assert len(chunks) == 2
    
    assert chunks[0].section == "a"
    assert chunks[0].parent == "CONFIDENTIALITY"
    assert chunks[0].hierarchy_path == ["CONFIDENTIALITY", "a"]
    
    assert chunks[1].section == "Section 1"
    assert chunks[1].parent == "ARTICLE IV"
    assert chunks[1].hierarchy_path == ["ARTICLE IV", "Section 1"]
