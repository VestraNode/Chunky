# Автори: Raimex Team · Подгорський Костянтин Олександрович
# Authors: Raimex Team · Podhorskyi Kostiantyn Oleksandrovych
# Сайт / Website: https://vn.raimex.space
# Репозиторій / Repository: https://github.com/VestraNode/Chunky

import argparse

import numpy as np
from PIL import Image

from stego.SChunky import decoder as s_dec, encoder as s_enc
from stego.chunky11 import decoder as c_dec, encoder as c_enc

METHODS = {"schunky": (s_enc, s_dec), "chunky11": (c_enc, c_dec)}


def hide(image_path, text, out_path, method="chunky11", key=None, chunk=8, limit=40):
    encoder, _ = METHODS[method]
    image = np.array(Image.open(image_path).convert("RGB"))
    result, report = encoder.encode_with_report(
        image, chunk, limit, text.encode("utf-8"), use_key=key is not None, key=key or 0)
    Image.fromarray(result).save(out_path, format="PNG")
    return report


def reveal(image_path, method="chunky11", key=None):
    _, decoder = METHODS[method]
    image = np.array(Image.open(image_path).convert("RGB"))
    data, report = decoder.decode_with_report(image, use_key=key is not None, key=key or 0)
    return data.decode("utf-8"), report


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    enc = sub.add_parser("hide")
    enc.add_argument("image")
    enc.add_argument("text")
    enc.add_argument("output")
    dec = sub.add_parser("reveal")
    dec.add_argument("image")
    for p in (enc, dec):
        p.add_argument("--method", choices=sorted(METHODS), default="chunky11")
        p.add_argument("--key", type=int, default=None)
    args = parser.parse_args()
    if args.command == "hide":
        report = hide(args.image, args.text, args.output, args.method, args.key)
        print("chunks used:", report["chunks_used"], "of", report["chunks_total"])
    else:
        text, _ = reveal(args.image, args.method, args.key)
        print(text)


if __name__ == "__main__":
    main()
