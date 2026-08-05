#!/usr/bin/env python3
"""
Generates a neofetch/terminal-style profile card (light_mode.svg / dark_mode.svg)
for the GitHub profile README: ASCII art of the GitHub avatar on the left,
live account stats on the right.

Requires:
  - GH_TOKEN      : GitHub token with `read:user` + `repo` scopes (Actions secret)
  - GH_USERNAME   : GitHub username to profile (defaults to "amandewatnitrr")

Image source: drop a square, high-contrast photo at scripts/avatar.(png|jpg) and
it's used instead of the (low-res) GitHub avatar. Delete it to fall back to the
live avatar fetch.

Run: python scripts/generate_profile_card.py
Outputs: profile_card_light.svg, profile_card_dark.svg (repo root)
"""

import os
import sys
import io
import glob
import datetime
import requests
from PIL import Image, ImageOps, ImageFilter

USERNAME = os.environ.get("GH_USERNAME", "amandewatnitrr")
TOKEN = os.environ.get("GH_TOKEN")

HEADERS = {"Authorization": f"bearer {TOKEN}"} if TOKEN else {}
REST_HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

ACCENT = "#0A66C2"
DOT_COLS = 60          # grid columns for the halftone portrait
DOT_CELL = 8            # px spacing between dot centers
DOT_MAX_RADIUS = 3.9    # px, darkest pixel -> this radius (cell/2 minus a hairline gap)
LOCAL_IMAGE_GLOB = os.path.join(os.path.dirname(__file__), "avatar.*")


# ---------- data fetching ----------

def fetch_user():
    r = requests.get(f"https://api.github.com/users/{USERNAME}", headers=REST_HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


def graphql(query, variables=None):
    r = requests.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": variables or {}},
        headers=HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(data["errors"])
    return data["data"]


def fetch_repo_stats():
    """Sum stargazers + top languages by byte size across owned, non-fork repos."""
    query = """
    query($login: String!, $after: String) {
      user(login: $login) {
        repositories(first: 100, after: $after, ownerAffiliations: OWNER, isFork: false) {
          totalCount
          pageInfo { hasNextPage endCursor }
          nodes {
            stargazerCount
            languages(first: 5, orderBy: {field: SIZE, direction: DESC}) {
              edges { size node { name } }
            }
          }
        }
      }
    }
    """
    stars = 0
    lang_bytes = {}
    total_repos = 0
    after = None
    while True:
        data = graphql(query, {"login": USERNAME, "after": after})
        repos = data["user"]["repositories"]
        total_repos = repos["totalCount"]
        for node in repos["nodes"]:
            stars += node["stargazerCount"]
            for edge in node["languages"]["edges"]:
                name = edge["node"]["name"]
                lang_bytes[name] = lang_bytes.get(name, 0) + edge["size"]
        if repos["pageInfo"]["hasNextPage"]:
            after = repos["pageInfo"]["endCursor"]
        else:
            break
    top_langs = sorted(lang_bytes.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return total_repos, stars, [name for name, _ in top_langs]


def fetch_total_commits(created_at):
    """contributionsCollection is capped at a 1-year window, so loop year by year."""
    start_year = datetime.datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").year
    end_year = datetime.datetime.utcnow().year
    query = """
    query($login: String!, $from: DateTime!, $to: DateTime!) {
      user(login: $login) {
        contributionsCollection(from: $from, to: $to) {
          totalCommitContributions
          restrictedContributionsCount
        }
      }
    }
    """
    total = 0
    for year in range(start_year, end_year + 1):
        frm = f"{year}-01-01T00:00:00Z"
        to = f"{year}-12-31T23:59:59Z"
        data = graphql(query, {"login": USERNAME, "from": frm, "to": to})
        cc = data["user"]["contributionsCollection"]
        total += cc["totalCommitContributions"] + cc["restrictedContributionsCount"]
    return total


# ---------- ascii art ----------

def load_source_image_bytes(fallback_url):
    matches = glob.glob(LOCAL_IMAGE_GLOB)
    if matches:
        print(f"using local image: {matches[0]}")
        with open(matches[0], "rb") as f:
            return f.read()
    return requests.get(fallback_url, timeout=20).content


def autocrop_to_subject(img, bust_fraction=0.55):
    """If the image has an alpha channel, crop tight to the non-transparent
    subject, then trim off the bottom (torso/shoulders) so the character
    budget below is spent mostly on the face, not empty space or fabric."""
    if img.mode != "RGBA":
        return img
    bbox = img.split()[3].getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    trimmed_bottom = top + int((bottom - top) * bust_fraction)
    return img.crop((left, top, right, trimmed_bottom))


def flatten_to_white(img):
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        flat = Image.new("RGB", img.size, "white")
        flat.paste(img, mask=img.split()[3])
        return flat
    return img.convert("RGB")


def image_to_dot_grid(image_bytes, cols=DOT_COLS):
    """Downsample to a brightness grid for a halftone dot portrait.

    Text-glyph ASCII art looks fine rendered at full size locally, but a
    GitHub README shrinks the embedded image to fit its column width --
    at that display size, small <text> glyphs are smaller than a pixel and
    anti-alias into a flat blob, discarding all detail regardless of how
    good the source dithering was. Dots don't have that failure mode:
    a <circle> stays a crisp, correctly-sized shape at any scale, and its
    radius encodes brightness continuously (256 levels) instead of being
    bucketed into a handful of discrete character shapes.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = autocrop_to_subject(img)
    img = flatten_to_white(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)  # stretch contrast so mid-tones separate
    rows = max(1, round(cols * (img.height / img.width)))  # dot cells are square, no font-aspect correction needed
    # sharpen at full resolution, before downscale -- brings out eye sockets,
    # nose bridge, mouth line as edges instead of letting LANCZOS blur them
    # into the surrounding tone before there are enough cells to represent them
    img = img.filter(ImageFilter.UnsharpMask(radius=4, percent=180, threshold=2))
    img = img.resize((cols, rows), Image.LANCZOS)
    pixels = list(img.getdata())
    grid = [pixels[y * cols:(y + 1) * cols] for y in range(rows)]
    return grid


# ---------- svg rendering ----------

def xml_escape(s):
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_dots(grid, pad, fg_dark_dot):
    """One <circle> per grid cell, radius scaled by darkness. A light gamma
    curve (0.8) fattens mid-tone dots slightly so faces don't read as a
    faint smudge -- pure linear brightness->radius under-represents
    mid-grays at this dot density."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    circles = []
    for r in range(rows):
        for c in range(cols):
            brightness = grid[r][c]
            darkness = (1 - brightness / 255) ** 0.8
            radius = darkness * DOT_MAX_RADIUS
            if radius < 0.35:
                continue
            cx = pad + c * DOT_CELL + DOT_CELL / 2
            cy = pad + r * DOT_CELL + DOT_CELL / 2
            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.2f}" fill="{fg_dark_dot}"/>'
            )
    dot_width = pad * 2 + cols * DOT_CELL
    dot_height = pad * 2 + rows * DOT_CELL
    return circles, dot_width, dot_height


def render_svg(dot_grid, stats_lines, theme):
    is_dark = theme == "dark"
    bg = "#0d1117" if is_dark else "#ffffff"
    fg = "#c9d1d9" if is_dark else "#24292f"
    dim = "#6e7681" if is_dark else "#57606a"

    pad = 24
    stats_font_size = 13
    stats_line_height = 16

    dot_circles, dot_block_width, dot_block_height = render_dots(dot_grid, pad, ACCENT)
    stats_x = dot_block_width + 40

    stats_block_height = pad + len(stats_lines) * stats_line_height
    height = max(dot_block_height, stats_block_height) + 14
    width = int(stats_x + 560)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Fira Code, Consolas, Menlo, monospace">',
        f'<rect width="100%" height="100%" fill="{bg}" rx="10"/>',
    ]
    svg.extend(dot_circles)

    y = pad
    for line in stats_lines:
        if line.startswith("§"):  # header marker
            content = line[1:]
            svg.append(
                f'<text x="{stats_x}" y="{y}" font-size="{stats_font_size}" fill="{ACCENT}" '
                f'font-weight="bold">{xml_escape(content)}</text>'
            )
        elif ":" in line:
            label, _, value = line.partition(":")
            svg.append(
                f'<text x="{stats_x}" y="{y}" font-size="{stats_font_size}">'
                f'<tspan fill="{dim}">{xml_escape(label)}:</tspan> '
                f'<tspan fill="{fg}">{xml_escape(value.strip())}</tspan></text>'
            )
        else:
            svg.append(
                f'<text x="{stats_x}" y="{y}" font-size="{stats_font_size}" fill="{fg}">{xml_escape(line)}</text>'
            )
        y += stats_line_height

    svg.append("</svg>")
    return "\n".join(svg)


# ---------- main ----------

def main():
    if not TOKEN:
        sys.exit("GH_TOKEN env var is required (a GitHub PAT with read:user + repo scopes).")
    user = fetch_user()
    image_bytes = load_source_image_bytes(user["avatar_url"])
    dot_grid = image_to_dot_grid(image_bytes)

    total_repos, stars, top_langs = fetch_repo_stats()
    total_commits = fetch_total_commits(user["created_at"])

    account_age_days = (datetime.datetime.utcnow() - datetime.datetime.strptime(
        user["created_at"], "%Y-%m-%dT%H:%M:%SZ")).days
    years, rem_days = divmod(account_age_days, 365)
    months = rem_days // 30

    stats_lines = [
        f"§{USERNAME}@github  " + "-" * 40,
        f"Account Age: {years}y {months}m",
        f"Followers: {user['followers']}",
        f"Following: {user['following']}",
        "",
        f"Top Languages: {', '.join(top_langs) if top_langs else 'n/a'}",
        "",
        f"§GitHub Stats  " + "-" * 43,
        f"Public Repos: {total_repos}",
        f"Total Stars: {stars}",
        f"Total Commits: {total_commits}",
        "",
        "§Find me  " + "-" * 48,
        "GitHub: github.com/" + USERNAME,
        "LinkedIn: linkedin.com/in/aman-kumar-dewangan-akd13o1",
        "LeetCode: leetcode.com/u/" + USERNAME,
    ]

    for theme in ("light", "dark"):
        svg = render_svg(dot_grid, stats_lines, theme)
        out_path = f"profile_card_{theme}.svg"
        with open(out_path, "w") as f:
            f.write(svg)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()