import hashlib, html, json, os, re, sys, urllib.parse, urllib.request, concurrent.futures as cf
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


def search(q, n=24, extra=""):
    from ddgs import DDGS
    kw = {"size": "Large"} if extra == "large" else {}
    return [x["image"] for x in DDGS().images(q, max_results=n, **kw)]


def dl(u, folder):
    try:
        name = hashlib.md5(u.encode()).hexdigest()[:10]
        data = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=15).read()
        p = os.path.join(folder, name)
        open(p, "wb").write(data)
        im = Image.open(p)
        w, h = im.size
        ext = (im.format or "img").lower()
        if min(w, h) < 300:
            os.remove(p)
            return None
        os.rename(p, p + "." + ext)
        return p + "." + ext, w, h
    except Exception:
        return None


def sheet(files, out, T=220):
    cols = 6
    rows = (len(files) + cols - 1) // cols
    from PIL import ImageDraw
    S = Image.new("RGB", (cols * T, rows * (T + 16)), (40, 40, 40))
    d = ImageDraw.Draw(S)
    for i, (f, w, h) in enumerate(files):
        try:
            im = Image.open(f).convert("RGBA")
            im.thumbnail((T, T))
            bg = Image.new("RGBA", im.size, (255, 0, 255, 255)); bg.alpha_composite(im)
            x, y = (i % cols) * T, (i // cols) * (T + 16)
            S.paste(bg.convert("RGB"), (x, y))
            d.text((x + 2, y + T + 2), f"{i} {os.path.basename(f)[:10]} {w}x{h}", fill=(255, 255, 0))
        except Exception:
            pass
    S.save(out)


if __name__ == "__main__":
    tag, q = sys.argv[1], sys.argv[2]
    extra = sys.argv[3] if len(sys.argv) > 3 else ""
    folder = f"/tmp/aot/web/{tag}"
    os.makedirs(folder, exist_ok=True)
    urls = search(q, 30, extra)
    with cf.ThreadPoolExecutor(12) as ex:
        res = [r for r in ex.map(lambda u: dl(u, folder), urls) if r]
    json.dump({os.path.basename(r[0]): u for r, u in zip(res, urls)}, open(folder + "/urls.json", "w"))
    sheet(res, f"/tmp/aot/web/{tag}.png")
    print(tag, len(urls), "urls", len(res), "ok")
