#!/usr/bin/env python3
"""Clean raw Genius-scraped Mitski lyrics into plain lyric text.

The raw scrape (data/raw/mitski_lyrics.json) stores each song's lyrics as
whatever Genius put in the lyrics container, which includes scraper noise
glued onto the front of the actual lyrics:

    "49 ContributorsTranslationsRomânăSlovenščina...Abbey Lyrics[Verse 1]\\n..."

This script strips that noise for every song and writes a cleaned JSON file
with the same shape, ready for text analysis.
"""

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone

READ_MORE = "Read More"


def strip_header(title: str, lyrics: str) -> str:
    """Remove the contributor/translation header and any Genius annotation
    text that precedes the actual lyrics, returning just the lyrics body."""
    marker = f"{title} Lyrics"
    idx = lyrics.find(marker)
    if idx == -1:
        # Marker missing (unexpected format) - fall back to the raw text.
        rest = lyrics
    else:
        rest = lyrics[idx + len(marker) :]

    read_more_idx = rest.find(READ_MORE)
    first_bracket = rest.find("[")

    if read_more_idx != -1 and (first_bracket == -1 or read_more_idx < first_bracket):
        rest = rest[read_more_idx + len(READ_MORE) :]
    elif first_bracket > 0:
        rest = rest[first_bracket:]

    return rest


def normalize_whitespace(text: str) -> str:
    """Collapse scraper whitespace artifacts (nbsp, em-spaces, etc.) without
    touching newlines, then trim each line and the overall text."""
    normalized = "".join(
        " " if ch != "\n" and unicodedata.category(ch) == "Zs" else ch for ch in text
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def strip_section_tags(text: str) -> str:
    """Remove [Verse 1], [Chorus], etc. section markers and blank the result
    of doing so down to clean lyric lines."""
    without_tags = re.sub(r"^\[.*?\]$", "", text, flags=re.MULTILINE)
    lines = [line.strip() for line in without_tags.split("\n")]
    return "\n".join(line for line in lines if line)


def clean_song(song: dict, strip_tags: bool, min_length: int) -> dict:
    body = strip_header(song["title"], song["lyrics"])
    body = normalize_whitespace(body)
    if strip_tags:
        body = strip_section_tags(body)

    if len(body) < min_length:
        print(
            f"warning: '{song['title']}' cleaned to only {len(body)} chars",
            file=sys.stderr,
        )

    return {
        "title": song["title"],
        "url": song["url"],
        "lyrics": body,
        "line_count": body.count("\n") + 1 if body else 0,
        "word_count": len(body.split()),
    }


def clean_lyrics_file(data: dict, strip_tags: bool, min_length: int) -> dict:
    cleaned_songs = [clean_song(song, strip_tags, min_length) for song in data["songs"]]
    return {
        "artist": data.get("artist"),
        "source_url": data.get("source_url"),
        "scraped_at": data.get("scraped_at"),
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "song_count": len(cleaned_songs),
        "songs": cleaned_songs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/raw/mitski_lyrics.json",
        help="Path to the raw scraped lyrics JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/mitski_lyrics_clean.json",
        help="Path to write the cleaned lyrics JSON file.",
    )
    parser.add_argument(
        "--strip-section-tags",
        action="store_true",
        help="Also remove [Verse]/[Chorus]-style section markers.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Warn if a cleaned song's lyrics are shorter than this many characters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    cleaned = clean_lyrics_file(data, args.strip_section_tags, args.min_length)

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Cleaned {cleaned['song_count']} songs -> {args.output}")


if __name__ == "__main__":
    main()
