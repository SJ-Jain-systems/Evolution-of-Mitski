"""Unit tests for the lexical-metric functions.

Run with:  python -m pytest tests/  (or: python tests/test_text.py)
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import text as T


def test_tokenize_keeps_contractions():
    assert T.tokenize("I don't know, I'm fine") == ["i", "don't", "know", "i'm", "fine"]


def test_tokenize_folds_curly_apostrophe():
    # Curly and straight apostrophes must tokenize identically.
    assert T.tokenize("I don’t") == T.tokenize("I don't") == ["i", "don't"]


def test_ttr_bounds():
    assert T.type_token_ratio([]) == 0.0
    assert T.type_token_ratio(["a", "a", "a"]) == 1 / 3
    assert T.type_token_ratio(["a", "b", "c"]) == 1.0


def test_mattr_falls_back_to_ttr_when_short():
    toks = ["a", "b", "a", "c"]
    assert T.mattr(toks, window=50) == T.type_token_ratio(toks)


def test_mattr_windowed_value():
    # Two windows of ["a","b"] and ["b","c"] over window=2 -> each TTR 1.0.
    toks = ["a", "b", "c"]
    assert T.mattr(toks, window=2) == 1.0
    # A repeated window pulls the average below 1.
    toks2 = ["a", "a", "b"]
    # windows: ["a","a"]=0.5, ["a","b"]=1.0 -> mean 0.75
    assert abs(T.mattr(toks2, window=2) - 0.75) < 1e-9


def test_hapax_ratio():
    # vocab {a,b,c}; a appears twice, b and c once -> 2/3 hapax.
    assert abs(T.hapax_ratio(["a", "a", "b", "c"]) - 2 / 3) < 1e-9


def test_pronoun_counts():
    counts = T.pronoun_counts("I love you and you love us")
    assert counts["first_singular"] == 1  # "I"
    assert counts["second"] == 2  # "you" x2
    assert counts["first_plural"] == 1  # "us"


def test_motif_counts():
    counts = T.motif_counts("blood and bones and water and water")
    assert counts["body"] == 2  # blood, bones
    assert counts["water"] == 2  # water x2


def test_lexical_summary_keys():
    s = T.lexical_summary("a b a b c")
    assert set(s) >= {
        "tokens",
        "types",
        "ttr",
        "mattr",
        "guiraud_r",
        "mtld",
        "yules_k",
        "hapax_ratio",
        "mean_word_length",
        "line_count",
        "mean_line_length",
        "refrain_ratio",
    }
    assert s["tokens"] == 5 and s["types"] == 3


# --- new diversity / structure / valence metrics ---------------------------


def test_yules_k_zero_for_all_unique():
    # No word repeats -> concentration K is 0.
    assert abs(T.yules_k(["a", "b", "c", "d"])) < 1e-9
    # A heavily repeated text concentrates -> K well above 0.
    assert T.yules_k(["a"] * 10 + ["b"]) > T.yules_k(list("abcdefghij"))
    assert T.yules_k([]) == 0.0


def test_mtld_higher_for_more_varied_text():
    varied = (
        "the quick brown fox jumps over the lazy dog and runs far beyond every green hill"
    ).split()
    repetitive = ["yeah", "yeah", "no", "no"] * 6
    assert T.mtld(varied) > T.mtld(repetitive)
    assert T.mtld([]) == 0.0


def test_split_lines_drops_blanks():
    assert T.split_lines("one\n\n  \ntwo\n") == ["one", "two"]


def test_line_summary_refrain_ratio_counts_repeats():
    text = "hold me now\nhold me now\nlet me go\nhold me now"
    s = T.line_summary(text)
    assert s["line_count"] == 4
    # Two of the four lines echo an earlier line (the first "hold me now" does not).
    assert abs(s["refrain_ratio"] - 2 / 4) < 1e-9
    # "hold me now" and "let me go" are both three words.
    assert abs(s["mean_line_length"] - 3.0) < 1e-9


def test_line_summary_empty_is_safe():
    s = T.line_summary("   \n  \n")
    assert s == {"line_count": 0, "mean_line_length": 0.0, "refrain_ratio": 0.0}


def test_valence_stats_spread_and_range():
    lex = {"love": 3, "hate": -3, "ok": 1}
    s = T.valence_stats("love hate ok unscored", lex)
    assert s["valence_range"] == 6.0  # +3 to -3
    assert s["valence_std"] > 0
    assert 0 < s["valence_coverage"] < 1  # "unscored" is not in the lexicon


def test_valence_stats_empty_is_zeroed():
    s = T.valence_stats("nothing scored here", {"love": 3})
    assert s == {
        "valence_mean": 0.0,
        "valence_std": 0.0,
        "valence_range": 0.0,
        "valence_coverage": 0.0,
    }


# --- distinctive words -----------------------------------------------------


def test_distinctive_words_favours_target_only_terms():
    # "moon" appears only in the target; "night" is shared evenly. The target-
    # only content word should rank above the evenly shared one.
    target = Counter({"moon": 6, "night": 4, "and": 20})
    background = Counter({"night": 4, "and": 20, "sun": 6})
    ranked = T.distinctive_words(target, background, n=5, stopwords={"and"})
    words = [w for w, _ in ranked]
    assert "moon" in words
    assert words[0] == "moon"
    # "night" is shared evenly, so it scores near zero and below "moon".
    score = dict(ranked)
    assert score["moon"] > score.get("night", -1)


def test_distinctive_words_respects_stopwords_and_min_count():
    target = Counter({"moon": 5, "the": 100, "rare": 1})
    background = Counter({"other": 50})
    ranked = T.distinctive_words(target, background, n=10, stopwords={"the"}, min_count=2)
    words = [w for w, _ in ranked]
    assert "the" not in words  # stopword dropped
    assert "rare" not in words  # below min_count
    assert "moon" in words


def test_distinctive_words_empty_inputs_are_safe():
    assert T.distinctive_words(Counter(), Counter({"a": 3})) == []
    assert T.distinctive_words(Counter({"a": 3}), Counter()) == []


# --- valence ---------------------------------------------------------------


def test_mean_valence_sign():
    lex = {"love": 3, "good": 3, "hate": -3, "die": -3}
    assert T.mean_valence("love and good", lex) > 0
    assert T.mean_valence("hate and die", lex) < 0


def test_mean_valence_ignores_unscored_and_empty():
    lex = {"love": 3}
    # Only "love" is scored; the mean of a single +3 hit is 3.0.
    assert T.mean_valence("love the quiet morning", lex) == 3.0
    assert T.mean_valence("", lex) == 0.0
    assert T.mean_valence("no scored words here", lex) == 0.0


def test_valence_lexicon_loads_and_scores_corpus_words():
    lex = T.load_valence_lexicon()
    assert len(lex) > 1000
    assert lex["love"] > 0 and lex["hate"] < 0


# --- rhyme (phonetics) -----------------------------------------------------


def test_rime_takes_last_stressed_vowel_to_end():
    # "light" = L AY1 T -> rime starts at the stressed AY1.
    assert T.rime(["L", "AY1", "T"]) == ("AY1", "T")
    # "eye" = AY1 and "sky" = S K AY1 share the rime AY1, so they rhyme.
    assert T.rime(["AY1"]) == T.rime(["S", "K", "AY1"]) == ("AY1",)
    # Different rime -> no rhyme (light vs eye).
    assert T.rime(["L", "AY1", "T"]) != T.rime(["AY1"])


def test_rime_fallbacks():
    assert T.rime([]) == ()
    # No stressed vowel: fall back to the last vowel of any stress.
    assert T.rime(["B", "AH0", "T"]) == ("AH0", "T")
    # No vowel at all: return the whole sequence.
    assert T.rime(["S", "T"]) == ("S", "T")


def test_rhyme_summary_detects_end_rhymes():
    lex = {
        "sky": ("S", "K", "AY1"),
        "eye": ("AY1",),
        "cry": ("K", "R", "AY1"),
        "here": ("HH", "IH1", "R"),
    }
    # Three of four line-ends rhyme (sky/eye/cry on AY1); "here" does not.
    text = "up in the sky\ni caught your eye\ni did not cry\nbut i am here"
    s = T.rhyme_summary(text, lex, window=4)
    assert abs(s["rhyme_coverage"] - 1.0) < 1e-9  # all four end-words are known
    assert abs(s["rhyme_density"] - 3 / 4) < 1e-9
    assert s["end_rhyme_pairs_per_line"] > 0


def test_rhyme_summary_skips_section_tags_and_is_safe_when_empty():
    lex = {"eye": ("AY1",), "sky": ("S", "K", "AY1")}
    # The [Chorus] tag line must not count as a rhyme-bearing line.
    s = T.rhyme_summary("[Chorus]\nin your eye\nin the sky", lex, window=4)
    assert abs(s["rhyme_coverage"] - 1.0) < 1e-9
    assert abs(s["rhyme_density"] - 1.0) < 1e-9
    empty = T.rhyme_summary("", lex)
    assert empty == {"rhyme_density": 0.0, "end_rhyme_pairs_per_line": 0.0, "rhyme_coverage": 0.0}


def test_pronunciation_lexicon_loads_corpus_words():
    lex = T.load_pronunciation_lexicon()
    assert len(lex) > 1000
    # A word certain to be in the corpus and in cmudict, with a plausible rime.
    assert lex["eye"] == ("AY1",)
    assert T.rime(lex["eye"]) == ("AY1",)


# --- song structure (sections) ---------------------------------------------


def test_canonical_section_maps_tags():
    assert T._canonical_section("Verse 1") == "verse"
    assert T._canonical_section("Chorus") == "chorus"
    # Pre-/post-chorus must not collapse into "chorus".
    assert T._canonical_section("Pre-Chorus") == "prechorus"
    assert T._canonical_section("Post-Chorus") == "postchorus"
    # The singer after a colon is ignored.
    assert T._canonical_section("Bridge: Mitski") == "bridge"
    assert T._canonical_section("Spoken Word") == "other"


def test_parse_sections_splits_on_tags_and_keeps_lead():
    text = "a lead line\n[Verse 1]\nfirst\nsecond\n[Chorus]\nhook line"
    sections = T.parse_sections(text)
    # Implicit leading block, then verse, then chorus.
    assert [t for t, _ in sections] == ["other", "verse", "chorus"]
    assert sections[1][1] == ["first", "second"]


def test_section_summary_counts_and_shares():
    text = "[Verse 1]\nline one\nline two\n[Chorus]\nhook\nhook again\n[Bridge]\nturn"
    s = T.section_summary(text)
    assert s["section_count"] == 3.0
    assert s["has_bridge"] == 1.0
    assert s["distinct_section_types"] == 3.0
    # 2 of the 5 lyric lines are in the chorus.
    assert abs(s["chorus_share"] - 2 / 5) < 1e-9


def test_section_summary_detects_repeated_blocks():
    # The same chorus block appears twice -> half the two content blocks repeat.
    text = "[Chorus]\nhold on\nlet go\n[Chorus]\nhold on\nlet go"
    s = T.section_summary(text)
    assert abs(s["section_repetition"] - 0.5) < 1e-9


def test_section_summary_empty_is_zeroed():
    s = T.section_summary("   \n  ")
    assert s == {
        "section_count": 0.0,
        "distinct_section_types": 0.0,
        "chorus_share": 0.0,
        "has_bridge": 0.0,
        "section_repetition": 0.0,
    }


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
