"""Figure builders for the report. Each ``fig_*`` function takes the album (and
sometimes song) table and returns a matplotlib Figure, so the Quarto document
and ``scripts/make_figures.py`` render exactly the same charts.

Design follows the data-viz skill: fixed-order categorical colour assigned by
entity, a single-hue blue ramp for magnitude, thin marks, recessive grey
chrome, direct labels over legends where it reads cleanly, and one axis per
chart (never a dual y-axis).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from . import theme as TH

SHORT = {
    "Lush": "Lush",
    "Retired from Sad, New Career in Business": "Retired\nfrom Sad",
    "Bury Me at Makeout Creek": "Bury Me at\nMakeout Creek",
    "Puberty 2": "Puberty 2",
    "Be the Cowboy": "Be the\nCowboy",
    "Laurel Hell": "Laurel Hell",
    "The Land Is Inhospitable and So Are We": "The Land Is\nInhospitable",
}
SHORT_INLINE = {k: v.replace("\n", " ") for k, v in SHORT.items()}


def _fit_line(x, y):
    """Return (xs, ys, r) for a simple OLS fit."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    r = np.corrcoef(x, y)[0, 1]
    return xs, a + b * xs, r


# --------------------------------------------------------------------------- #
# 1. The headline thesis: words per minute over time
# --------------------------------------------------------------------------- #
def fig_wpm_over_time(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.0)
    x = albums["release_date"]
    y = albums["words_per_minute"]

    xs, ys, r = _fit_line(x.map(pd.Timestamp.toordinal), y)
    xs_dates = [pd.Timestamp.fromordinal(int(v)) for v in xs]
    ax.plot(xs_dates, ys, color=TH.MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)

    ax.plot(x, y, color=TH.BLUE, lw=2.2, zorder=2)
    ax.scatter(x, y, s=70, color=TH.BLUE, zorder=3, edgecolor=TH.SURFACE, linewidth=1.5)

    for _, row in albums.iterrows():
        ax.annotate(
            SHORT[row["album"]],
            (row["release_date"], row["words_per_minute"]),
            textcoords="offset points",
            xytext=(0, 12 if row["words_per_minute"] >= 55 else -26),
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TH.INK_2,
            linespacing=0.95,
        )
        ax.annotate(
            f"{row['words_per_minute']:.0f}",
            (row["release_date"], row["words_per_minute"]),
            textcoords="offset points",
            xytext=(0, 0),
            ha="center",
            va="center",
            fontsize=0,  # value carried by the y-axis; keep marks clean
        )

    ax.set_title("Mitski is saying less, minute for minute")
    ax.set_ylabel("Words per minute  (album lyrics ÷ runtime)")
    ax.set_ylim(40, 75)
    ax.annotate(
        f"Trend across career:  r = {r:.2f}",
        xy=(0.98, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9.5,
        color=TH.MUTED,
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. The mirror: wordiness vs. lexical diversity, two stacked panels
# --------------------------------------------------------------------------- #
def fig_wpm_vs_diversity_panels(albums: pd.DataFrame):
    TH.apply()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    for ax in (ax1, ax2):
        TH.style_axes(ax)

    x = albums["release_date"]

    ax1.plot(x, albums["words_per_minute"], color=TH.BLUE, lw=2.2)
    ax1.scatter(
        x,
        albums["words_per_minute"],
        s=55,
        color=TH.BLUE,
        edgecolor=TH.SURFACE,
        linewidth=1.4,
        zorder=3,
    )
    ax1.set_ylabel("Words per minute")
    ax1.set_title("As the words thin out, each word does more work")
    ax1.annotate(
        "fewer words per minute →",
        xy=(0.02, 0.10),
        xycoords="axes fraction",
        fontsize=9,
        color=TH.MUTED,
    )

    # Diversity: the video's own metric is total/unique (repetition); we plot
    # its inverse-reading companion, MATTR, so "up" means "more varied".
    ax2.plot(x, albums["mattr"], color=TH.ORANGE, lw=2.2)
    ax2.scatter(
        x, albums["mattr"], s=55, color=TH.ORANGE, edgecolor=TH.SURFACE, linewidth=1.4, zorder=3
    )
    ax2.set_ylabel("Lexical diversity\n(MATTR, length-robust)")
    ax2.set_xlabel("Album release")
    ax2.annotate(
        "more varied vocabulary →",
        xy=(0.02, 0.88),
        xycoords="axes fraction",
        fontsize=9,
        color=TH.MUTED,
    )

    # Mark the two extremes the story turns on, without labelling every point
    # (album identities are established in the words-per-minute chart above).
    extremes = albums.sort_values("mattr")
    for _, row in pd.concat([extremes.head(1), extremes.tail(1)]).iterrows():
        ax2.annotate(
            SHORT_INLINE[row["album"]],
            (row["release_date"], row["mattr"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.2,
            color=TH.INK_2,
        )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 3. The inverse relationship as a scatter (video's literal metric)
# --------------------------------------------------------------------------- #
def fig_inverse_scatter(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.6)
    x = albums["words_per_minute"]
    y = albums["repetition_index"]

    # Colour early vs mature era by entity role (two categorical slots).
    mature = albums["album_no"] >= 4
    ax.scatter(
        x[~mature],
        y[~mature],
        s=90,
        color=TH.CATEGORICAL[4],
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Early era (2012–14)",
    )
    ax.scatter(
        x[mature],
        y[mature],
        s=90,
        color=TH.BLUE,
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Mature era (2016–23)",
    )

    xs, ys, r = _fit_line(x[mature], y[mature])
    ax.plot(xs, ys, color=TH.BLUE, lw=1.4, ls=(0, (4, 3)), zorder=1)

    for _, row in albums.iterrows():
        ax.annotate(
            SHORT_INLINE[row["album"]],
            (row["words_per_minute"], row["repetition_index"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8.4,
            color=TH.INK_2,
        )

    ax.set_xlabel("Words per minute  →  (denser)")
    ax.set_ylabel("Repetition index  (total ÷ unique words)  →  (more repetitive)")
    ax.set_title("Denser albums repeat themselves; sparse albums reach wider")
    ax.legend(loc="upper left")
    ax.annotate(
        f"Mature-era fit:  r = {r:.2f}",
        xy=(0.98, 0.06),
        xycoords="axes fraction",
        ha="right",
        fontsize=9.5,
        color=TH.MUTED,
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. Words per song, by album (are the songs themselves getting sparser?)
# --------------------------------------------------------------------------- #
def fig_words_per_song(songs: pd.DataFrame, albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.2)
    order = albums.sort_values("release_date")["album"].tolist()
    rng = np.random.default_rng(7)
    for i, album in enumerate(order):
        vals = songs.loc[songs["album"] == album, "word_count"].values
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=34,
            color=TH.SEQUENTIAL[3],
            edgecolor=TH.SURFACE,
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )
        ax.plot([i - 0.28, i + 0.28], [vals.mean(), vals.mean()], color=TH.INK, lw=2.2, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[a] for a in order], fontsize=8.2, linespacing=0.95)
    ax.set_ylabel("Words per song")
    ax.set_title("Per-song word counts, album by album  (black bar = album mean)")
    return fig


# --------------------------------------------------------------------------- #
# 5. Pronoun mix: I / you / we across the discography
# --------------------------------------------------------------------------- #
def fig_pronoun_mix(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.0)
    order = albums.sort_values("release_date")
    idx = np.arange(len(order))
    i_share = order["pron_first_singular_share"].values
    you = order["pron_second_share"].values
    we = order["pron_first_plural_share"].values

    colors = [TH.BLUE, TH.CATEGORICAL[3], TH.CATEGORICAL[5]]
    labels = ["I / me / my", "you / your", "we / us / our"]
    bottom = np.zeros(len(order))
    for share, c, lab in zip([i_share, you, we], colors, labels):
        ax.bar(
            idx,
            share,
            bottom=bottom,
            width=0.66,
            color=c,
            label=lab,
            edgecolor=TH.SURFACE,
            linewidth=1.5,
        )
        bottom = bottom + share
    ax.set_xticks(idx)
    ax.set_xticklabels([SHORT[a] for a in order["album"]], fontsize=8.2, linespacing=0.95)
    ax.set_ylabel("Share of personal pronouns")
    ax.set_ylim(0, 1)
    ax.set_title("Who the songs address: the private “I”, the “you”, and the collective “we”")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", visible=True)
    return fig


# --------------------------------------------------------------------------- #
# 6. Motif heatmap: recurring imagery rates by album
# --------------------------------------------------------------------------- #
def fig_motif_heatmap(albums: pd.DataFrame):
    motifs = ["body", "water", "fire_light", "home_domestic", "death", "animals"]
    nice = ["Body", "Water", "Fire / light", "Home / domestic", "Death", "Animals"]
    order = albums.sort_values("release_date")
    M = np.array([[row[f"motif_{m}_per_1k"] for m in motifs] for _, row in order.iterrows()])

    cmap = TH.sequential_cmap()
    TH.apply()
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    im = ax.imshow(M, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(motifs)))
    ax.set_xticklabels(nice, fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([SHORT_INLINE[a] for a in order["album"]], fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    vmax = M.max()
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=TH.INK if M[i, j] < 0.6 * vmax else TH.SURFACE,
            )
    ax.set_title("Recurring imagery, rate per 1,000 words")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 7. Emotional valence over time (AFINN word-level sentiment)
# --------------------------------------------------------------------------- #
def fig_valence_over_time(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.0)
    order = albums.sort_values("release_date")
    x = order["release_date"]
    y = order["mean_valence"]

    # A neutral reference so "above / below the line" reads as positive /
    # negative, and a single connecting line for the time trend.
    ax.axhline(0, color=TH.BASELINE, lw=1.2, zorder=1)
    ax.plot(x, y, color=TH.MUTED, lw=1.6, zorder=2)

    # Colour each point by sign: warm for negative, cool for positive. Two
    # categorical slots, assigned by meaning, not cycled.
    pos = y >= 0
    ax.scatter(x[pos], y[pos], s=80, color=TH.BLUE, edgecolor=TH.SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(
        x[~pos], y[~pos], s=80, color=TH.ORANGE, edgecolor=TH.SURFACE, linewidth=1.5, zorder=3
    )

    # Label above each point, except the two near the ceiling (which would
    # collide with the title), so no label crowds the x-axis.
    hi = y.max()
    for _, row in order.iterrows():
        near_top = row["mean_valence"] >= hi - 0.05
        ax.annotate(
            SHORT_INLINE[row["album"]],
            (row["release_date"], row["mean_valence"]),
            textcoords="offset points",
            xytext=(0, -16 if near_top else 10),
            ha="center",
            va="top" if near_top else "bottom",
            fontsize=8.2,
            color=TH.INK_2,
        )

    ax.set_ylabel("Mean word valence  (AFINN, negative ↔ positive)")
    ax.set_xlabel("Album release")
    ax.set_title("Average word sentiment barely tracks the mood of the records")
    ax.annotate(
        "more positive words →",
        xy=(0.02, 0.93),
        xycoords="axes fraction",
        fontsize=9,
        color=TH.MUTED,
    )
    ax.annotate(
        "more negative words →",
        xy=(0.02, 0.06),
        xycoords="axes fraction",
        fontsize=9,
        color=TH.MUTED,
    )
    return fig


# --------------------------------------------------------------------------- #
# 8. The visual "trilogy" colour story (qualitative, from the source video)
# --------------------------------------------------------------------------- #
def fig_trilogy(albums: pd.DataFrame):
    TH.apply()
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.25)
    ax.axis("off")

    panels = [
        ("Be the Cowboy", TH.TRILOGY["Be the Cowboy"], "explosion of red\nstepping into a persona"),
        (
            "Laurel Hell",
            TH.TRILOGY["Laurel Hell"],
            "the same frame, in black\nthe persona meets fame",
        ),
        (
            "The Land Is\nInhospitable…",
            TH.TRILOGY["The Land Is Inhospitable and So Are We"],
            "the camera pulls back\n“what world made this?”",
        ),
    ]
    for i, (name, color, note) in enumerate(panels):
        ax.add_patch(plt.Rectangle((i + 0.08, 0.35), 0.84, 0.78, color=color, ec="none"))
        txt_color = "#ffffff" if name != "The Land Is\nInhospitable…" else TH.INK
        ax.text(
            i + 0.5,
            0.74,
            name,
            ha="center",
            va="center",
            color=txt_color,
            fontsize=10,
            fontweight="bold",
            linespacing=0.95,
        )
        ax.text(
            i + 0.5,
            0.20,
            note,
            ha="center",
            va="center",
            color=TH.INK_2,
            fontsize=8.4,
            linespacing=1.05,
        )
        if i < 2:
            arr = FancyArrowPatch(
                (i + 0.93, 0.74),
                (i + 1.07, 0.74),
                arrowstyle="-|>",
                mutation_scale=14,
                color=TH.MUTED,
                lw=1.6,
            )
            ax.add_patch(arr)
    ax.text(
        1.5,
        1.16,
        "One image, three times. Each album pulls the camera back",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=TH.INK,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 9. Refrain reliance vs. word repetition (line-level echo of the repetition
#    index): repeated lines are NOT a simple function of time -- they track how
#    repetitive the *words* are, so The Land, the least repetitive album, also
#    leans least on refrains.
# --------------------------------------------------------------------------- #
def fig_refrain_over_time(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.4)
    x = albums["repetition_index"]
    y = albums["refrain_ratio"] * 100

    xs, ys, r = _fit_line(x, y)
    ax.plot(xs, ys, color=TH.MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)

    mature = albums["album_no"] >= 4
    ax.scatter(
        x[~mature],
        y[~mature],
        s=90,
        color=TH.CATEGORICAL[4],
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Early era (2012–14)",
    )
    ax.scatter(
        x[mature],
        y[mature],
        s=90,
        color=TH.BLUE,
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Mature era (2016–23)",
    )
    for _, row in albums.iterrows():
        ax.annotate(
            SHORT_INLINE[row["album"]],
            (row["repetition_index"], row["refrain_ratio"] * 100),
            textcoords="offset points",
            xytext=(8, 3),
            fontsize=8.2,
            color=TH.INK_2,
        )

    ax.set_xlabel("Repetition index  (total ÷ unique words)  →  more repetitive")
    ax.set_ylabel("Repeated-line share  (% of lines that echo an earlier line)")
    ax.set_title("Repeated lines track word repetition, not the calendar")
    ax.legend(loc="lower right")
    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.02, 0.94),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TH.MUTED,
    )
    return fig


# --------------------------------------------------------------------------- #
# 11. End-rhyme density vs. word density: rhyme is a property of sound, and it
#     tracks how many words there are to rhyme (words per minute), not the
#     calendar -- the sparse late records rhyme less because there is less to
#     rhyme, the same shape as the refrain and repetition stories.
# --------------------------------------------------------------------------- #
def fig_rhyme_over_time(albums: pd.DataFrame):
    fig, ax = TH.new_fig(9.0, 5.4)
    x = albums["words_per_minute"]
    y = albums["rhyme_density"] * 100

    xs, ys, r = _fit_line(x, y)
    ax.plot(xs, ys, color=TH.MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)

    mature = albums["album_no"] >= 4
    ax.scatter(
        x[~mature],
        y[~mature],
        s=90,
        color=TH.CATEGORICAL[4],
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Early era (2012–14)",
    )
    ax.scatter(
        x[mature],
        y[mature],
        s=90,
        color=TH.BLUE,
        edgecolor=TH.SURFACE,
        linewidth=1.5,
        zorder=3,
        label="Mature era (2016–23)",
    )
    for _, row in albums.iterrows():
        ax.annotate(
            SHORT_INLINE[row["album"]],
            (row["words_per_minute"], row["rhyme_density"] * 100),
            textcoords="offset points",
            xytext=(8, 3),
            fontsize=8.2,
            color=TH.INK_2,
        )

    ax.set_xlabel("Words per minute  →  (denser)")
    ax.set_ylabel("End-rhyme density  (% of line-ends that rhyme with a neighbour)")
    ax.set_title("Rhyme tracks how much there is to rhyme, not the calendar")
    ax.legend(loc="lower right")
    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.02, 0.94),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=9.5,
        color=TH.MUTED,
    )
    return fig


# --------------------------------------------------------------------------- #
# 12. Song structure over time: mean sections per song and chorus share. Unlike
#     the words, the scaffolding barely moves -- the compression happens *within*
#     sections (shorter lines and songs), not by dropping the verse/chorus frame.
# --------------------------------------------------------------------------- #
def fig_structure_over_time(albums: pd.DataFrame):
    TH.apply()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)
    for ax in (ax1, ax2):
        TH.style_axes(ax)

    order = albums.sort_values("release_date")
    x = order["release_date"]

    ax1.plot(x, order["sec_section_count"], color=TH.BLUE, lw=2.2)
    ax1.scatter(
        x,
        order["sec_section_count"],
        s=55,
        color=TH.BLUE,
        edgecolor=TH.SURFACE,
        linewidth=1.4,
        zorder=3,
    )
    ax1.set_ylabel("Sections per song\n(mean)")
    ax1.set_ylim(0, order["sec_section_count"].max() + 1.5)
    ax1.set_title("The scaffolding holds steady while the words thin")

    ax2.plot(x, order["sec_chorus_share"] * 100, color=TH.ORANGE, lw=2.2)
    ax2.scatter(
        x,
        order["sec_chorus_share"] * 100,
        s=55,
        color=TH.ORANGE,
        edgecolor=TH.SURFACE,
        linewidth=1.4,
        zorder=3,
    )
    ax2.set_ylabel("Share of lines\nin a chorus (%)")
    ax2.set_ylim(0, (order["sec_chorus_share"] * 100).max() + 12)
    ax2.set_xlabel("Album release")

    # Direct-label the endpoints on the top panel so album identities are legible
    # without a legend, matching the other time-series charts.
    for _, row in pd.concat([order.head(1), order.tail(1)]).iterrows():
        ax1.annotate(
            SHORT_INLINE[row["album"]],
            (row["release_date"], row["sec_section_count"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8.2,
            color=TH.INK_2,
        )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 10. Vocabulary growth: the type-accumulation curve across the whole career
# --------------------------------------------------------------------------- #
def fig_vocab_growth(growth: pd.DataFrame, albums: pd.DataFrame):
    """Cumulative distinct words vs. cumulative total words, song by song in
    release order. ``growth`` is ``data.build_vocab_growth(...)``."""
    fig, ax = TH.new_fig(9.0, 5.4)
    order = albums.sort_values("release_date")["album"].tolist()
    color_by_album = {a: TH.CATEGORICAL[i % len(TH.CATEGORICAL)] for i, a in enumerate(order)}

    total = growth["cumulative_words"].to_numpy()
    types = growth["cumulative_types"].to_numpy()

    # One continuous curve, coloured by the album each song belongs to, so the
    # eye can see whether new vocabulary keeps arriving in the late records.
    for album in order:
        mask = growth["album"] == album
        ax.plot(
            total[mask],
            types[mask],
            color=color_by_album[album],
            lw=2.4,
            solid_capstyle="round",
            zorder=3,
        )
    ax.scatter(total, types, s=12, color=TH.INK, alpha=0.25, zorder=2)

    # Direct album labels at each album's last point, no legend.
    for album in order:
        sub = growth[growth["album"] == album]
        last = sub.iloc[-1]
        ax.annotate(
            SHORT_INLINE[album],
            (last["cumulative_words"], last["cumulative_types"]),
            textcoords="offset points",
            xytext=(6, -2),
            fontsize=7.8,
            color=color_by_album[album],
            va="center",
        )

    ax.set_xlabel("Cumulative words sung across the career")
    ax.set_ylabel("Cumulative distinct words")
    ax.set_title("Vocabulary keeps arriving: distinct words vs. total words, in release order")
    return fig
