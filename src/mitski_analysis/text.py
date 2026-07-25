"""Text metrics for the Mitski lyrics corpus.

Everything here operates on plain lyric strings (already cleaned of section
tags and scraper noise by ``scripts/clean_lyrics.py``). The functions are
deliberately dependency-free so they are easy to test and reason about.

Key idea: lexical diversity measured as a raw type-token ratio (TTR) is
*length dependent* -- a longer text mechanically repeats more common words and
so scores a lower TTR even if the writing is no less varied. Because Mitski's
albums differ in total word count, we lead with a length-robust measure
(MATTR, the moving-average type-token ratio) and report the raw TTR alongside
it so the video's original definition is still visible.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

# Word token: a run of letters, allowing internal straight apostrophes so
# contractions ("don't", "i'm") stay single tokens. Curly apostrophes are
# folded to straight ones in ``normalize`` before this pattern is applied.
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def normalize(text: str) -> str:
    """Lower-case and fold typographic apostrophes to a plain ASCII quote."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    return text.lower()


def tokenize(text: str) -> list[str]:
    """Return the ordered list of word tokens in ``text``."""
    return _WORD_RE.findall(normalize(text))


def type_token_ratio(tokens: Iterable[str]) -> float:
    """Raw TTR = unique tokens / total tokens (the video's definition,
    inverted so that *higher means more varied*)."""
    tokens = list(tokens)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio (Covington & McFall, 2010).

    Averages the TTR of every ``window``-length sliding window, which removes
    the length dependence of a single raw TTR. Falls back to a plain TTR when
    the text is shorter than one window.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return type_token_ratio(tokens)
    ratios = []
    for start in range(0, n - window + 1):
        chunk = tokens[start : start + window]
        ratios.append(len(set(chunk)) / window)
    return sum(ratios) / len(ratios)


def guiraud_r(tokens: list[str]) -> float:
    """Guiraud's root-TTR: types / sqrt(tokens). Less length-sensitive than
    raw TTR; grows with vocabulary richness."""
    n = len(tokens)
    if n == 0:
        return 0.0
    return len(set(tokens)) / math.sqrt(n)


def hapax_ratio(tokens: list[str]) -> float:
    """Fraction of the vocabulary that appears exactly once (hapax legomena).
    A high value signals reaching for many single-use words."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(counts)


def mean_word_length(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return sum(len(t) for t in tokens) / len(tokens)


def yules_k(tokens: list[str]) -> float:
    """Yule's K (Yule, 1944): a length-robust *concentration* measure.

    K is high when a text leans on a small set of words used often, low when
    vocabulary is spread thin. Unlike a raw TTR it does not drift with length,
    so it corroborates MATTR from the opposite direction (K falls as diversity
    rises). Defined as ``1e4 * (sum_i i^2 * V_i - N) / N^2`` where ``V_i`` is the
    number of word types occurring ``i`` times and ``N`` is the token count.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    freqs = Counter(tokens)
    # V_i: how many types occur exactly i times.
    spectrum = Counter(freqs.values())
    m2 = sum((i * i) * vi for i, vi in spectrum.items())
    return 1e4 * (m2 - n) / (n * n)


def mtld(tokens: list[str], threshold: float = 0.72) -> float:
    """Measure of Textual Lexical Diversity (McCarthy & Jarvis, 2010).

    MTLD is the mean length of the longest running-TTR "factors" that stay above
    ``threshold`` before resetting, averaged over a forward and a backward pass.
    It is the diversity measure most robust to text length, so it is the best
    single check on the length-sensitive raw TTR. Returns the token count when
    the text never drops below the threshold (maximally diverse for its length).
    """
    n = len(tokens)
    if n == 0:
        return 0.0

    def _one_pass(seq: list[str]) -> float:
        factors = 0.0
        types: set[str] = set()
        count = 0
        for tok in seq:
            count += 1
            types.add(tok)
            ttr = len(types) / count
            if ttr <= threshold:
                factors += 1.0
                types.clear()
                count = 0
        if count > 0:
            # Partial trailing factor, scaled by how far it fell toward the cut.
            ttr = len(types) / count
            factors += (1.0 - ttr) / (1.0 - threshold)
        return len(seq) / factors if factors > 0 else float(len(seq))

    return (_one_pass(tokens) + _one_pass(list(reversed(tokens)))) / 2.0


def split_lines(text: str) -> list[str]:
    """Non-empty, stripped lyric lines (blank lines and pure whitespace dropped)."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def line_summary(text: str) -> dict[str, float]:
    """Structural, line-level metrics for one lyric blob.

    ``mean_line_length`` is words per non-empty line; ``line_count`` is the number
    of such lines; ``refrain_ratio`` is the share of lines that repeat a line seen
    earlier (a transparent proxy for how much a lyric leans on a refrain -- it
    rises on the compressed, hook-driven late songs like *Working for the Knife*).
    """
    lines = split_lines(text)
    if not lines:
        return {"line_count": 0, "mean_line_length": 0.0, "refrain_ratio": 0.0}
    norm_lines = [" ".join(tokenize(ln)) for ln in lines]
    word_counts_per_line = [len(nl.split()) for nl in norm_lines]
    seen: set[str] = set()
    repeats = 0
    for nl in norm_lines:
        if not nl:
            continue
        if nl in seen:
            repeats += 1
        else:
            seen.add(nl)
    n_lines = len(lines)
    total_words = sum(word_counts_per_line) or 1
    return {
        "line_count": n_lines,
        "mean_line_length": total_words / n_lines,
        "refrain_ratio": repeats / n_lines,
    }


def lexical_summary(text: str, window: int = 50) -> dict[str, float]:
    """All scalar lexical metrics for one text blob, in one pass."""
    tokens = tokenize(text)
    summary = {
        "tokens": len(tokens),
        "types": len(set(tokens)),
        "ttr": type_token_ratio(tokens),
        "mattr": mattr(tokens, window=window),
        "guiraud_r": guiraud_r(tokens),
        "mtld": mtld(tokens),
        "yules_k": yules_k(tokens),
        "hapax_ratio": hapax_ratio(tokens),
        "mean_word_length": mean_word_length(tokens),
    }
    summary.update(line_summary(text))
    return summary


# --- Thematic lexicons -----------------------------------------------------
# Small, transparent keyword sets used to trace recurring imagery across the
# discography. These are intentionally hand-curated and conservative; they are
# a lens on the lyrics, not a claim of exhaustive coverage.
MOTIF_LEXICONS: dict[str, set[str]] = {
    "body": {
        "body",
        "bodies",
        "blood",
        "bone",
        "bones",
        "skin",
        "heart",
        "hearts",
        "hand",
        "hands",
        "eye",
        "eyes",
        "mouth",
        "lips",
        "hair",
        "chest",
        "knee",
        "knees",
        "arms",
        "arm",
        "face",
        "teeth",
        "veins",
        "breath",
    },
    "water": {
        "water",
        "waters",
        "sea",
        "ocean",
        "wave",
        "waves",
        "rain",
        "river",
        "lake",
        "tears",
        "tear",
        "drown",
        "drowning",
        "swim",
        "flood",
        "wet",
        "creek",
        "tide",
        "pool",
    },
    "fire_light": {
        "fire",
        "flame",
        "flames",
        "burn",
        "burning",
        "burns",
        "light",
        "lights",
        "spark",
        "sun",
        "star",
        "stars",
        "glow",
        "lightning",
        "fireworks",
        "shine",
        "bright",
        "smoke",
        "ember",
    },
    "home_domestic": {
        "home",
        "house",
        "room",
        "door",
        "kitchen",
        "bed",
        "wall",
        "walls",
        "lamp",
        "window",
        "floor",
        "table",
        "husband",
        "wife",
        "married",
        "marry",
        "phone",
        "car",
    },
    "death": {
        "die",
        "died",
        "dying",
        "death",
        "dead",
        "grave",
        "bury",
        "buried",
        "ghost",
        "kill",
        "gone",
        "goodbye",
        "funeral",
        "coffin",
        "heaven",
    },
    "animals": {
        "dog",
        "dogs",
        "horse",
        "cat",
        "cats",
        "bird",
        "birds",
        "wolf",
        "coyote",
        "buffalo",
        "bug",
        "angel",
        "pearl",
        "cowboy",
    },
}

# First / second / first-plural person pronoun sets. These trace the shift the
# source video highlights: private "I", the addressed "you", and the collective
# "we" that surfaces by *The Land Is Inhospitable and So Are We*.
PRONOUNS: dict[str, set[str]] = {
    "first_singular": {"i", "i'm", "i'll", "i've", "i'd", "me", "my", "mine", "myself"},
    "second": {"you", "you're", "you'll", "you've", "you'd", "your", "yours", "yourself"},
    "first_plural": {"we", "we're", "we'll", "we've", "us", "our", "ours", "ourselves"},
}


def motif_counts(text: str) -> dict[str, int]:
    """Raw hit count for each motif lexicon in ``text``."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    return {motif: sum(counts[w] for w in words) for motif, words in MOTIF_LEXICONS.items()}


def pronoun_counts(text: str) -> dict[str, int]:
    """Raw pronoun-group counts in ``text``."""
    tokens = tokenize(text)
    counts = Counter(tokens)
    return {group: sum(counts[w] for w in words) for group, words in PRONOUNS.items()}


# --- Distinctive words (keyness) -------------------------------------------
# Which words most set one album apart from the rest of the discography? Raw
# frequency answers the wrong question (it just surfaces "the", "you", "and"),
# and a plain ratio over-rewards words that appear once or twice. The weighted
# log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn,
# 2008) is the standard fix: it compares a word's rate in the target text to its
# rate in a background corpus, smooths both toward the pooled distribution, and
# divides by an estimated standard error so rare-word noise is damped. The
# result is a z-like score; positive means "more characteristic of the target".

# A small, transparent function-word list so the distinctive-word ranking
# surfaces content, not grammar. Hand-curated in the same spirit as the motif
# lexicons: short, legible, and easy to audit rather than an exhaustive list.
# It also drops two kinds of transcription noise that would otherwise rank as
# "distinctive": song-structure tags that survive cleaning ("[Verse: Mitski &
# Choir]" tokenizes to verse/mitski/choir) and sung vocables ("doo", "ra").
STOPWORDS: set[str] = {
    # function words
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "i'm",
    "i'll",
    "i've",
    "i'd",
    "if",
    "in",
    "is",
    "it",
    "it's",
    "its",
    "me",
    "my",
    "no",
    "not",
    "of",
    "oh",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "was",
    "we",
    "we're",
    "were",
    "what",
    "what'd",
    "when",
    "who",
    "will",
    "with",
    "would",
    "you",
    "you're",
    "you'll",
    "you've",
    "you'd",
    "your",
    "yours",
    "am",
    "are",
    "'cause",
    "cause",
    "just",
    "all",
    "like",
    "get",
    "got",
    "can",
    "could",
    "now",
    "yeah",
    "where",
    "don't",
    "gonna",
    "wanna",
    "gotta",
    "again",
    "more",
    "only",
    "very",
    "really",
    # structural transcription tags (survive cleaning as bare tokens)
    "verse",
    "chorus",
    "prechorus",
    "pre",
    "post",
    "bridge",
    "intro",
    "outro",
    "hook",
    "refrain",
    "interlude",
    "mitski",
    "choir",
    "prod",
    "feat",
    "ft",
    # sung vocables
    "la",
    "na",
    "ooh",
    "ah",
    "mm",
    "doo",
    "ra",
    "da",
    "dum",
    "ba",
    "hey",
    "woah",
    "whoa",
    "oo",
    "uh",
    "ooo",
    "ahh",
}


def word_counts(text: str) -> Counter:
    """Token frequency map for ``text`` (reuses the shared tokenizer)."""
    return Counter(tokenize(text))


def distinctive_words(
    target_counts: Counter,
    background_counts: Counter,
    n: int = 10,
    stopwords: set[str] | None = STOPWORDS,
    min_count: int = 2,
) -> list[tuple[str, float]]:
    """Return the ``n`` words most distinctive of ``target_counts`` relative to
    ``background_counts``, scored by weighted log-odds with an informative
    Dirichlet prior (Monroe, Colaresi & Quinn 2008).

    ``background_counts`` is the pooled rest of the corpus (the target may or may
    not be included in it; the prior handles both). Words in ``stopwords`` and
    words occurring fewer than ``min_count`` times in the target are dropped
    before ranking, so the result reads as content the album is *about*.
    """
    if stopwords is None:
        stopwords = set()

    # Pooled counts form the Dirichlet prior alpha_w; alpha_0 is its total.
    pooled = Counter()
    pooled.update(target_counts)
    pooled.update(background_counts)
    alpha0 = sum(pooled.values())
    n_target = sum(target_counts.values())
    n_bg = sum(background_counts.values())
    if n_target == 0 or n_bg == 0 or alpha0 == 0:
        return []

    scores: list[tuple[str, float]] = []
    for word, y_t in target_counts.items():
        if word in stopwords or y_t < min_count:
            continue
        a_w = pooled[word]  # prior pseudo-count for this word
        y_b = background_counts.get(word, 0)
        # Smoothed log-odds in target vs. background.
        num_t = y_t + a_w
        num_b = y_b + a_w
        den_t = n_target + alpha0 - num_t
        den_b = n_bg + alpha0 - num_b
        delta = math.log(num_t / den_t) - math.log(num_b / den_b)
        var = 1.0 / num_t + 1.0 / num_b
        scores.append((word, delta / math.sqrt(var)))

    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:n]


# --- Valence (sentiment) ---------------------------------------------------
# Mean emotional valence per lyric, from a bundled word->score lexicon (AFINN).
# The lexicon is a flat, human-readable file under ``data/lexicons/`` so the
# scoring stays transparent and fully offline, matching the project's use of
# small, auditable lexicons elsewhere.


def load_valence_lexicon(root: Path | None = None) -> dict[str, int]:
    """Load the bundled AFINN valence lexicon (``word<TAB>score`` per line).

    Comment lines (starting with ``#``) and blanks are skipped. ``root`` is the
    repo root; when omitted the lexicon is located relative to this file.
    """
    if root is None:
        # src/mitski_analysis/text.py -> repo root is two parents up.
        root = Path(__file__).resolve().parents[2]
    path = Path(root) / "data" / "lexicons" / "valence_afinn.txt"
    lexicon: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            word, _, score = line.partition("\t")
            word = word.strip()
            score = score.strip()
            if word and score:
                lexicon[normalize(word)] = int(score)
    return lexicon


def mean_valence(text: str, lexicon: dict[str, int]) -> float:
    """Average valence over the tokens of ``text`` that appear in ``lexicon``.

    Returns 0.0 when no token is scored (or the text is empty), so an album with
    no lexicon hits reads as neutral rather than undefined.
    """
    tokens = tokenize(text)
    scored = [lexicon[t] for t in tokens if t in lexicon]
    if not scored:
        return 0.0
    return sum(scored) / len(scored)


def valence_stats(text: str, lexicon: dict[str, int]) -> dict[str, float]:
    """Mean, spread and range of word-level valence over the scored tokens.

    The report's point is that *mean* valence misses Mitski's meaning; the spread
    (``valence_std``) and ``valence_range`` add the missing dimension -- how far a
    lyric swings between its brightest and darkest words, not just where it
    averages. ``valence_coverage`` is the share of tokens the lexicon scored at
    all, a caveat on how much of the text the sentiment number even sees. All
    fields are 0.0 when no token is scored.
    """
    tokens = tokenize(text)
    scored = [lexicon[t] for t in tokens if t in lexicon]
    if not scored:
        return {
            "valence_mean": 0.0,
            "valence_std": 0.0,
            "valence_range": 0.0,
            "valence_coverage": 0.0,
        }
    mean = sum(scored) / len(scored)
    var = sum((s - mean) ** 2 for s in scored) / len(scored)
    return {
        "valence_mean": mean,
        "valence_std": math.sqrt(var),
        "valence_range": float(max(scored) - min(scored)),
        "valence_coverage": len(scored) / len(tokens) if tokens else 0.0,
    }


# --- Rhyme (phonetics) -----------------------------------------------------
# How densely does an album rhyme, and does the rhyming thin out as the lyrics
# get sparser? Rhyme is a property of *sound*, not spelling ("love"/"move" look
# alike but do not rhyme; "eye"/"I" rhyme but share no letters), so this needs
# pronunciations. We bundle a subset of the CMU Pronouncing Dictionary (ARPABET
# phonemes, restricted to the corpus vocabulary) under ``data/lexicons/`` in the
# same flat, offline, auditable form as the AFINN lexicon.

# ARPABET vowel phonemes (without the trailing 0/1/2 stress digit). Every rhyme
# hangs off a vowel, so these are what ``rime`` searches back from.
_ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
)

# A line that is only a section tag ("[Chorus]", "[Verse 2]") is not a lyric line
# and must not contribute a rhyme word.
_SECTION_TAG_RE = re.compile(r"^\[.*\]$")


def _is_vowel(phoneme: str) -> bool:
    """True if ``phoneme`` is an ARPABET vowel (ignoring its stress digit)."""
    return phoneme.rstrip("012") in _ARPABET_VOWELS


def rime(phonemes: Iterable[str]) -> tuple[str, ...]:
    """The rhyming tail of a pronunciation: from the last *stressed* vowel to the
    end (the linguistic "rime"). Two words rhyme iff their rimes are equal.

    Falls back to the last vowel of any stress when no vowel carries primary or
    secondary stress, and to the whole sequence when there is no vowel at all.
    Stress digits are kept so that, e.g., stressed and unstressed endings do not
    collapse together.
    """
    phones = list(phonemes)
    if not phones:
        return ()
    stressed = [i for i, p in enumerate(phones) if p.endswith(("1", "2")) and _is_vowel(p)]
    if stressed:
        start = stressed[-1]
    else:
        vowels = [i for i, p in enumerate(phones) if _is_vowel(p)]
        start = vowels[-1] if vowels else 0
    return tuple(phones[start:])


def load_pronunciation_lexicon(root: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Load the bundled CMU pronunciation subset (``word<TAB>PHONEMES`` per line).

    Values are the ARPABET phoneme tuple for the word's primary pronunciation.
    Comment lines (starting with ``#``) and blanks are skipped. ``root`` is the
    repo root; when omitted the lexicon is located relative to this file, exactly
    like :func:`load_valence_lexicon`.
    """
    if root is None:
        # src/mitski_analysis/text.py -> repo root is two parents up.
        root = Path(__file__).resolve().parents[2]
    path = Path(root) / "data" / "lexicons" / "pronunciations_cmudict.txt"
    lexicon: dict[str, tuple[str, ...]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            word, _, phones = line.partition("\t")
            word = word.strip()
            phones = phones.strip()
            if word and phones:
                lexicon[normalize(word)] = tuple(phones.split())
    return lexicon


def _line_final_words(text: str) -> list[str]:
    """Line-final word token of every lyric line, skipping section-tag lines and
    lines with no word token. Used as the rhyme-bearing positions."""
    finals = []
    for line in split_lines(text):
        if _SECTION_TAG_RE.match(line):
            continue
        toks = tokenize(line)
        if toks:
            finals.append(toks[-1])
    return finals


def rhyme_summary(
    text: str, lexicon: dict[str, tuple[str, ...]] | None = None, window: int = 4
) -> dict[str, float]:
    """Rhyme metrics over the line-final words of ``text``.

    A line rhymes when another line whose end-word falls within ``window`` lines
    shares its rime (see :func:`rime`). Returns:

    - ``rhyme_density``: share of *covered* lines (those whose end-word is in the
      pronunciation lexicon) that rhyme with another nearby line.
    - ``end_rhyme_pairs_per_line``: count of within-window rhyming line pairs,
      normalized by the number of lyric lines -- a density that also rewards
      lines rhyming with several neighbours.
    - ``rhyme_coverage``: share of lyric lines whose end-word the lexicon knows, a
      caveat on how much of the rhyme picture is even visible (mirrors
      ``valence_coverage``).

    All fields are 0.0 when there is nothing to measure.
    """
    if lexicon is None:
        lexicon = load_pronunciation_lexicon()
    finals = _line_final_words(text)
    n_lines = len(finals)
    if n_lines == 0:
        return {"rhyme_density": 0.0, "end_rhyme_pairs_per_line": 0.0, "rhyme_coverage": 0.0}

    rimes: list[tuple[str, ...] | None] = [
        rime(lexicon[w]) if w in lexicon else None for w in finals
    ]
    covered = [i for i, r in enumerate(rimes) if r]
    coverage = len(covered) / n_lines
    if len(covered) < 2:
        return {
            "rhyme_density": 0.0,
            "end_rhyme_pairs_per_line": 0.0,
            "rhyme_coverage": coverage,
        }

    rhyming_lines: set[int] = set()
    pairs = 0
    for a_pos, i in enumerate(covered):
        for j in covered[a_pos + 1 :]:
            if j - i > window:
                break
            if rimes[i] == rimes[j]:
                pairs += 1
                rhyming_lines.add(i)
                rhyming_lines.add(j)
    return {
        "rhyme_density": len(rhyming_lines) / len(covered),
        "end_rhyme_pairs_per_line": pairs / n_lines,
        "rhyme_coverage": coverage,
    }


# --- Song structure (sections) ---------------------------------------------
# The cleaned lyrics keep the transcription's section markers ("[Verse 1]",
# "[Chorus]", "[Bridge]") on their own lines. They are noise for the token
# metrics (and dropped by STOPWORDS there), but they are exactly the signal for
# reading *structure*: how many sections a song has, how much of it is chorus,
# and whether it repeats whole blocks. This traces the source video's arc from a
# different angle -- does the compression that thins the words also change the
# scaffolding of the songs?

_SECTION_RE = re.compile(r"^\[(.*?)\]\s*$")

# Coarse section types the raw tags map onto. Order matters: the pre-/post-chorus
# checks must run before the plain "chorus" check so "[Pre-Chorus]" does not read
# as a chorus. Anything unrecognised is "other".
_SECTION_ALIASES: tuple[tuple[str, str], ...] = (
    ("prechorus", "prechorus"),
    ("pre-chorus", "prechorus"),
    ("pre chorus", "prechorus"),
    ("postchorus", "postchorus"),
    ("post-chorus", "postchorus"),
    ("post chorus", "postchorus"),
    ("chorus", "chorus"),
    ("verse", "verse"),
    ("bridge", "bridge"),
    ("intro", "intro"),
    ("outro", "outro"),
    ("refrain", "refrain"),
    ("hook", "hook"),
    ("interlude", "interlude"),
    ("instrumental", "instrumental"),
)


def _canonical_section(label: str) -> str:
    """Map a raw section-tag label ("Verse 1", "Pre-Chorus", "Chorus: Mitski")
    to a coarse type. Only the part before any ``:`` (which names singers, not the
    section) is inspected."""
    head = label.split(":", 1)[0].strip().lower()
    for needle, canonical in _SECTION_ALIASES:
        if needle in head:
            return canonical
    return "other"


def parse_sections(text: str) -> list[tuple[str, list[str]]]:
    """Split a lyric into ``(section_type, lines)`` blocks on its ``[...]`` tags.

    Non-empty lyric lines are accumulated under the most recent tag; content that
    appears before the first tag becomes an implicit leading ``other`` section, so
    no lyric line is ever dropped. A tag with no lines under it (e.g. an
    ``[Instrumental Break]``) still yields an empty block, which keeps the section
    *count* faithful to the transcription.
    """
    sections: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_type is not None or current_lines:
            sections.append((current_type or "other", current_lines))

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SECTION_RE.match(line)
        if m:
            _flush()
            current_type = _canonical_section(m.group(1))
            current_lines = []
        else:
            current_lines.append(line)
    _flush()
    return sections


def section_summary(text: str) -> dict[str, float]:
    """Structural metrics for one song from its section tags.

    - ``section_count``: number of tagged (or implicit) blocks.
    - ``distinct_section_types``: how many *kinds* of section appear.
    - ``chorus_share``: share of lyric lines that sit in a chorus or post-chorus.
    - ``has_bridge``: 1.0 if the song has a bridge, else 0.0.
    - ``section_repetition``: share of non-empty blocks whose text repeats an
      earlier block verbatim -- a whole-section analogue of ``refrain_ratio``.

    All fields are 0.0 for a lyric with no lines.
    """
    sections = parse_sections(text)
    total_lines = sum(len(lines) for _, lines in sections)
    if total_lines == 0:
        return {
            "section_count": 0.0,
            "distinct_section_types": 0.0,
            "chorus_share": 0.0,
            "has_bridge": 0.0,
            "section_repetition": 0.0,
        }

    chorus_lines = sum(len(lines) for stype, lines in sections if stype in ("chorus", "postchorus"))
    types = {stype for stype, _ in sections}

    seen: set[str] = set()
    repeated_blocks = 0
    content_blocks = 0
    for _stype, lines in sections:
        if not lines:
            continue
        content_blocks += 1
        key = "\n".join(" ".join(tokenize(ln)) for ln in lines)
        if key in seen:
            repeated_blocks += 1
        else:
            seen.add(key)

    return {
        "section_count": float(len(sections)),
        "distinct_section_types": float(len(types)),
        "chorus_share": chorus_lines / total_lines,
        "has_bridge": 1.0 if "bridge" in types else 0.0,
        "section_repetition": repeated_blocks / content_blocks if content_blocks else 0.0,
    }
