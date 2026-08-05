#!/usr/bin/env python
"""Render the Chinese Markdown report to PDF.

No pandoc, wkhtmltopdf, weasyprint or headless browser is available on this
machine, so the route is Markdown -> HTML -> LibreOffice -> PDF. LibreOffice's
HTML import is the weak link: it ignores most CSS, so the stylesheet stays
deliberately plain and the layout leans on things it does honour (table borders,
explicit image widths, font-family).

Images are rewritten to absolute paths -- LibreOffice resolves them against its
own working directory, not the HTML file's.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 1.6cm; }
body { font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback", sans-serif;
       font-size: 10.5pt; line-height: 1.5; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 14.5pt; margin-top: 22px; border-bottom: 1px solid #999; padding-bottom: 3px; }
h3 { font-size: 12pt; margin-top: 16px; }
h4 { font-size: 11pt; margin-top: 12px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #888; padding: 4px 7px; text-align: left; }
th { background: #e8e8e8; font-weight: bold; }
img { max-width: 100%; }
code { font-family: "Noto Mono", monospace; font-size: 9pt; background: #f0f0f0; padding: 1px 3px; }
pre { background: #f5f5f5; border: 1px solid #ccc; padding: 8px; font-size: 8.5pt;
      white-space: pre-wrap; word-wrap: break-word; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; color: #444; }
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", default=None, help="Output PDF (default: alongside the markdown)")
    ap.add_argument("--img-width", type=int, default=640, help="Pixel width forced on every image")
    args = ap.parse_args()

    md_path = Path(args.md).resolve()
    base = md_path.parent
    out_pdf = Path(args.out).resolve() if args.out else md_path.with_suffix(".pdf")

    text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])

    # LibreOffice resolves image src against its own cwd, and it honours neither
    # the CSS max-width nor the width attribute reliably -- a 2226 px figure
    # simply runs off the page. Physically downscale each image to a width that
    # fits the printable area (letter minus LibreOffice's default ~2 cm margins
    # is about 6.9 in, i.e. ~660 px at the 96 dpi it assumes) and point at the
    # resized copy, so the size is settled before it ever reaches the importer.
    from PIL import Image

    img_dir = out_pdf.parent / "_pdf_img"
    img_dir.mkdir(exist_ok=True)

    def fix_img(m):
        src = m.group(2)
        if src.startswith(("http://", "https://")):
            return m.group(0)
        p = Path(src) if src.startswith("/") else (base / src)
        p = p.resolve()
        if not p.exists():
            return f'{m.group(1)}"{p}"'
        try:
            im = Image.open(p).convert("RGB")
            if im.width > args.img_width:
                im = im.resize((args.img_width, round(im.height * args.img_width / im.width)),
                               Image.LANCZOS)
            dst = img_dir / f"{p.stem}.png"
            # Without explicit DPI metadata LibreOffice assumes 72, so a 640 px
            # figure lands 8.9 in wide and runs off the page. Stating 96 pins it.
            im.save(dst, dpi=(96, 96))
            return f'{m.group(1)}"{dst}"'
        except Exception:  # noqa: BLE001 - fall back to the original on any decode issue
            return f'{m.group(1)}"{p}"'

    html_body = re.sub(r'(<img[^>]*?src=)"([^"]+)"', fix_img, html_body)

    html = (f'<html><head><meta charset="utf-8"><style>{CSS}</style></head>'
            f"<body>{html_body}</body></html>")
    html_path = out_pdf.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    missing = [m for m in re.findall(r'<img[^>]*?src="([^"]+)"', html_body) if not Path(m).exists()]
    if missing:
        print("WARNING: images not found:", *missing, sep="\n  ", file=sys.stderr)

    r = subprocess.run(
        ["libreoffice", "--headless", "--convert-to", "pdf",
         "--outdir", str(out_pdf.parent), str(html_path)],
        capture_output=True, text=True, timeout=600,
    )
    produced = html_path.with_suffix(".pdf")
    if not produced.exists():
        print("LibreOffice failed:", r.stdout, r.stderr, file=sys.stderr)
        raise SystemExit(1)
    if produced != out_pdf:
        produced.rename(out_pdf)
    print(f"-> {out_pdf}  ({out_pdf.stat().st_size / 1e6:.1f} MB)")
    html_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
