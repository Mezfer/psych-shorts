"""Weekly review of log.csv. Uses ONLY the numbers you typed in from YouTube Studio; it never guesses.
Run from GitHub Actions (weekly_report.yml): the table appears in the job summary."""
import csv, sys
from collections import defaultdict

COLS = ["views", "viewed_vs_swiped_pct", "avg_view_pct", "likes", "comments", "subs_gained"]


def num(x):
    try:
        return float(str(x).replace("%", "").replace(",", ".").strip())
    except Exception:
        return None


def main():
    try:
        with open("log.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print("No log.csv yet.")
        return
    filled = [r for r in rows if any(num(r.get(c)) is not None for c in COLS)]
    print(f"## Weekly review\n\nVideos logged: {len(rows)} | with results filled in: {len(filled)}\n")
    if len(filled) < 8:
        print("Not enough data yet. Fill views / viewed_vs_swiped_pct / avg_view_pct / subs_gained in log.csv "
              "(from YouTube Studio) for at least 8 videos. Until then the formats are picked at random.")
        return
    for key in ("format", "cta_style", "subject"):
        groups = defaultdict(list)
        for r in filled:
            if r.get(key):
                groups[r[key]].append(r)
        if not groups:
            continue
        print(f"\n### By {key}\n\n| {key} | videos | " + " | ".join(COLS) + " |\n|---|---|" + "---|" * len(COLS))
        for g, rs in sorted(groups.items()):
            cells = []
            for c in COLS:
                v = [num(r.get(c)) for r in rs if num(r.get(c)) is not None]
                cells.append(f"{sum(v) / len(v):.1f}" if v else "-")
            print(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    metric = "viewed_vs_swiped_pct" if sum(num(r.get("viewed_vs_swiped_pct")) is not None for r in filled) >= 8 else "views"
    ranked = sorted((r for r in filled if num(r.get(metric)) is not None), key=lambda r: -num(r[metric]))
    print(f"\n### Best and worst by {metric}\n")
    for r in ranked[:3]:
        print(f"- BEST {r[metric]}: {r['title']} ({r.get('format')}, hook: {r.get('hook')})")
    for r in ranked[-3:]:
        print(f"- WORST {r[metric]}: {r['title']} ({r.get('format')}, hook: {r.get('hook')})")
    print("\nSmall numbers are noisy: treat a pattern as real only when several videos agree.")


if __name__ == "__main__":
    sys.exit(main())
