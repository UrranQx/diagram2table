"""Quick check script: convert SVG -> PNG using Wand (ImageMagick).

Usage:
    python check_svg_support.py --input data3/diagram.svg --output out.png --dpi 100
"""

import argparse
import sys
from pathlib import Path

try:
    from wand.image import Image as WandImage
except Exception as e:
    print("Wand import failed. Ensure 'wand' is installed and ImageMagick is available in PATH.")
    raise


def convert_svg_to_png(input_path: Path, output_path: Path, dpi: int = 100) -> None:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with WandImage(filename=str(input_path), resolution=dpi) as img:
        img.format = "png"
        img.save(filename=str(output_path))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input SVG file")
    parser.add_argument("--output", "-o", default="output.png", help="Output PNG file")
    parser.add_argument("--dpi", type=int, default=100, help="Rasterization DPI (default 100)")
    args = parser.parse_args(argv)

    try:
        convert_svg_to_png(Path(args.input), Path(args.output), dpi=args.dpi)
        print(f"Converted {args.input} -> {args.output} at {args.dpi} DPI")
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()