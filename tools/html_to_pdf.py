"""Render an HTML file to PDF with Edge (headless).

usage: python html_to_pdf.py report.html report.pdf
"""
import subprocess
import sys
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def main(html, pdf):
    url = Path(html).resolve().as_uri()
    subprocess.run([EDGE, "--headless=new", "--disable-gpu",
                    "--no-pdf-header-footer", f"--print-to-pdf={Path(pdf).resolve()}", url],
                   check=False)
    print("pdf:", pdf, Path(pdf).stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
