#!/usr/bin/env python3
"""Scrape Mitski lyrics using the Genius API and genius.com song pages into a JSON file."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

GENIUS_API_BASE = "https://api.genius.com"
DEFAULT_ARTIST_NAME = "Mitski"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; MitskiLyricScraper/1.0; "
    "+https://github.com/local/mitski-lyric-scraper)"
)
LYRICS_CONTAINER_MARKER = 'data-lyrics-container="true"'


@dataclass(frozen=True)
class Song:
    """A discovered song and its scraped lyrics."""

    title: str
    url: str
    lyrics: str


@dataclass(frozen=True)
class ScrapeError:
    """A song page that could not be scraped."""

    title: str
    url: str
    error: str


def fetch_html(url: str, timeout: float, user_agent: str) -> str:
    """Fetch a page and return its HTML, raising for HTTP errors."""

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def genius_api_get(
    path: str, access_token: str, timeout: float, params: dict | None = None
) -> dict:
    """Call the Genius API and return the decoded JSON body."""

    url = f"{GENIUS_API_BASE}{path}"
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"Authorization": f"Bearer {access_token}"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 401:
            raise RuntimeError(
                "Genius API rejected the access token (401 Unauthorized). "
                "Check --access-token / GENIUS_ACCESS_TOKEN."
            ) from exc
        raise


def find_artist(artist_name: str, access_token: str, timeout: float) -> tuple[int, str, str | None]:
    """Look up a Genius artist id, canonical name, and profile URL by name."""

    data = genius_api_get("/search", access_token, timeout, params={"q": artist_name})
    hits = data.get("response", {}).get("hits", [])
    candidates = [hit["result"]["primary_artist"] for hit in hits if "result" in hit]
    for candidate in candidates:
        if candidate.get("name", "").strip().lower() == artist_name.strip().lower():
            return candidate["id"], candidate["name"], candidate.get("url")
    if candidates:
        candidate = candidates[0]
        return candidate["id"], candidate["name"], candidate.get("url")
    raise RuntimeError(f"No Genius artist found for {artist_name!r}")


def discover_songs(
    artist_id: int,
    access_token: str,
    timeout: float,
    include_features: bool,
) -> list[tuple[str, str]]:
    """Return `(title, url)` pairs for every song credited to a Genius artist."""

    songs: list[tuple[str, str]] = []
    seen_ids: set[int] = set()
    page: int | None = 1
    while page is not None:
        data = genius_api_get(
            f"/artists/{artist_id}/songs",
            access_token,
            timeout,
            params={"per_page": 50, "page": page, "sort": "title"},
        )
        response = data.get("response", {})
        for song in response.get("songs", []):
            if song["id"] in seen_ids:
                continue
            if not include_features and song.get("primary_artist", {}).get("id") != artist_id:
                continue
            seen_ids.add(song["id"])
            songs.append((song["title"], song["url"]))
        page = response.get("next_page")
    return songs


def _extract_lyrics_fragments(song_html: str) -> list[str]:
    """Return the raw inner HTML of every Genius `data-lyrics-container` div."""

    tag_pattern = re.compile(r"<(/?)div\b[^>]*>", re.IGNORECASE)
    fragments: list[str] = []
    search_pos = 0
    while True:
        marker_index = song_html.find(LYRICS_CONTAINER_MARKER, search_pos)
        if marker_index == -1:
            break
        div_start = song_html.rfind("<div", 0, marker_index)
        opening_end = song_html.find(">", marker_index) if div_start != -1 else -1
        if div_start == -1 or opening_end == -1:
            search_pos = marker_index + len(LYRICS_CONTAINER_MARKER)
            continue

        depth = 1
        content_end = None
        cursor = opening_end + 1
        for match in tag_pattern.finditer(song_html, cursor):
            depth += -1 if match.group(1) == "/" else 1
            if depth == 0:
                content_end = match.start()
                cursor = match.end()
                break
        if content_end is None:
            break

        fragments.append(song_html[opening_end + 1 : content_end])
        search_pos = cursor
    return fragments


def _strip_tags(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    lines = [html.unescape(line).strip() for line in fragment.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_lyrics(song_html: str) -> str:
    """Extract lyric text from a genius.com song page."""

    fragments = _extract_lyrics_fragments(song_html)
    if not fragments:
        raise ValueError("Could not find Genius lyrics container(s)")

    lyrics = _strip_tags("\n".join(fragments))
    if not lyrics:
        raise ValueError("Genius lyrics container(s) were empty")
    return lyrics


def scrape_songs(
    artist_name: str,
    access_token: str,
    delay: float,
    timeout: float,
    user_agent: str,
    limit: int | None = None,
    fail_fast: bool = False,
    include_features: bool = False,
) -> tuple[list[Song], list[ScrapeError], str, str | None]:
    """Discover and scrape every Genius song credited to an artist."""

    artist_id, resolved_name, artist_url = find_artist(artist_name, access_token, timeout)
    song_links = discover_songs(artist_id, access_token, timeout, include_features)
    if not song_links:
        raise RuntimeError(
            f"Genius returned no songs for {resolved_name!r} (artist id {artist_id})."
        )
    if limit is not None:
        song_links = song_links[:limit]

    songs: list[Song] = []
    errors: list[ScrapeError] = []
    for index, (title, url) in enumerate(song_links, start=1):
        print(f"[{index}/{len(song_links)}] Scraping {title}: {url}")
        try:
            song_html = fetch_html(url, timeout, user_agent)
            songs.append(Song(title=title, url=url, lyrics=extract_lyrics(song_html)))
        except Exception as exc:
            if fail_fast:
                raise

            message = str(exc)
            errors.append(ScrapeError(title=title, url=url, error=message))
            print(f"  Skipping {title}: {message}")

        if index < len(song_links):
            time.sleep(delay)

    return songs, errors, resolved_name, artist_url


def write_output(
    songs: Iterable[Song],
    output_path: Path,
    artist_name: str,
    source_url: str | None,
    errors: Iterable[ScrapeError] = (),
) -> None:
    """Write scraped songs, failures, and metadata to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    song_data = [asdict(song) for song in songs]
    error_data = [asdict(error) for error in errors]
    payload = {
        "artist": artist_name,
        "source_url": source_url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "song_count": len(song_data),
        "error_count": len(error_data),
        "songs": song_data,
        "errors": error_data,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape lyrics for an artist from Genius.")
    parser.add_argument(
        "--artist", default=DEFAULT_ARTIST_NAME, help="Artist name to search for on Genius"
    )
    parser.add_argument(
        "--access-token",
        default=os.environ.get("GENIUS_ACCESS_TOKEN"),
        help="Genius API client access token (defaults to the GENIUS_ACCESS_TOKEN env var)",
    )
    parser.add_argument(
        "--output", default="mitski_lyrics.json", type=Path, help="JSON output path"
    )
    parser.add_argument(
        "--delay", default=1.0, type=float, help="Seconds to wait between song page requests"
    )
    parser.add_argument("--timeout", default=15.0, type=float, help="HTTP timeout in seconds")
    parser.add_argument(
        "--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent header for song pages"
    )
    parser.add_argument(
        "--limit", default=None, type=int, help="Optional maximum number of songs to scrape"
    )
    parser.add_argument(
        "--include-features",
        action="store_true",
        help="Also include songs where the artist is only a featured artist, not the primary one",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any song page cannot be scraped",
    )
    args = parser.parse_args()
    if not args.access_token:
        parser.error(
            "A Genius API access token is required: pass --access-token or set the "
            "GENIUS_ACCESS_TOKEN environment variable. Create one at https://genius.com/api-clients."
        )
    return args


def main() -> None:
    args = parse_args()
    songs, errors, resolved_name, source_url = scrape_songs(
        artist_name=args.artist,
        access_token=args.access_token,
        delay=args.delay,
        timeout=args.timeout,
        user_agent=args.user_agent,
        limit=args.limit,
        fail_fast=args.fail_fast,
        include_features=args.include_features,
    )
    write_output(songs, args.output, resolved_name, source_url, errors)
    print(f"Wrote {len(songs)} songs to {args.output}")
    if errors:
        print(f"Skipped {len(errors)} songs; see the output file's errors list for details")


if __name__ == "__main__":
    main()
