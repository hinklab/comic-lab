import os
import sys
import json
import time
import urllib.request
import urllib.error
import subprocess

def get_github_token():
    p = subprocess.Popen(['git', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate(input='protocol=https\nhost=github.com\n\n')
    for line in out.splitlines():
        if line.startswith('password='):
            return line.split('=', 1)[1]
    return None

def main():
    token = get_github_token()
    if not token:
        print("[ERROR] GitHub token topilmadi!")
        sys.exit(1)

    owner = "hinklab"
    repo = "comic-lab"
    tag = "v1.0.0"
    zip_path = os.path.abspath(r"dist\ComicLab_Portable_v1.0.0.zip")

    if not os.path.exists(zip_path):
        print(f"[ERROR] Zip fayl topilmadi: {zip_path}")
        sys.exit(1)

    file_size = os.path.getsize(zip_path)
    print(f"[INFO] Fayl topildi: {zip_path} ({file_size / (1024*1024):.2f} MB)")

    # 1. Create or get existing release
    releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "ComicLab-Release-Manager",
        "Accept": "application/vnd.github.v3+json"
    }

    # Check existing
    req = urllib.request.Request(releases_url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read().decode('utf-8'))

    target_release = None
    for r in releases:
        if r.get("tag_name") == tag:
            target_release = r
            print(f"[INFO] Mavjud release topildi: ID={r['id']}, Tag={tag}")
            break

    if not target_release:
        print(f"[INFO] Yangi release yaratilmoqda: {tag}...")
        release_payload = {
            "tag_name": tag,
            "target_commitish": "main",
            "name": "ComicLab Studio v1.0.0 (Windows Portable)",
            "body": (
                "## ComicLab Studio v1.0.0 (Windows Desktop Release)\n\n"
                "Komikslarni professional tarjima qilish, pufakchalarni avtomatik tozalash va shriftlarni terish (typesetting) studiyasi.\n\n"
                "### Asosiy Imkoniyatlar:\n"
                "- 100% Lossless HD Export (asl sifatli eksport)\n"
                "- Smart Bubble Detection & Erase (AI orqali matnlarni tozalash)\n"
                "- 60 FPS Accelerated Canvas (interaktiv tahrirlovchi)\n"
                "- Dynamic Comic Typography (Bangers, Inter, CC Wild Words va professional shriftlar)\n"
                "- Local In-Memory AI Cache va mustaqil oflayn arxitektura\n\n"
                "### O'rnatish:\n"
                "1. `ComicLab_Portable_v1.0.0.zip` faylini yuklab oling.\n"
                "2. Arxivni kompyuteringizdagi istalgan qulay papkaga oching (Extract).\n"
                "3. `ComicLab.exe` dasturini ishga tushiring.\n"
            ),
            "draft": False,
            "prerelease": False
        }
        create_req = urllib.request.Request(
            releases_url,
            data=json.dumps(release_payload).encode('utf-8'),
            headers={**headers, "Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(create_req) as resp:
            target_release = json.loads(resp.read().decode('utf-8'))
            print(f"[OK] Release muvaffaqiyatli yaratildi! ID={target_release['id']}")

    release_id = target_release["id"]

    # 2. Check if asset already uploaded, delete if exists to re-upload fresh
    assets = target_release.get("assets", [])
    for a in assets:
        if a.get("name") == "ComicLab_Portable_v1.0.0.zip":
            print(f"[INFO] Eski asset topildi (ID={a['id']}), o'chirilmoqda...")
            del_url = f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{a['id']}"
            del_req = urllib.request.Request(del_url, headers=headers, method="DELETE")
            with urllib.request.urlopen(del_req) as resp:
                print(f"[OK] Eski asset o'chirildi.")
            time.sleep(1)

    # 3. Upload asset to release
    upload_url = f"https://uploads.github.com/repos/{owner}/{repo}/releases/{release_id}/assets?name=ComicLab_Portable_v1.0.0.zip"
    print(f"[INFO] Arxiv yuklanmoqda ({file_size / (1024*1024):.2f} MB)... Bu 30-60 soniya vaqt olishi mumkin.")

    with open(zip_path, "rb") as fp:
        upload_req = urllib.request.Request(
            upload_url,
            data=fp,
            headers={
                "Authorization": f"token {token}",
                "User-Agent": "ComicLab-Release-Manager",
                "Content-Type": "application/zip",
                "Content-Length": str(file_size)
            },
            method="POST"
        )
        t0 = time.time()
        with urllib.request.urlopen(upload_req, timeout=300) as resp:
            asset_data = json.loads(resp.read().decode('utf-8'))
            elapsed = time.time() - t0
            print(f"[SUCCESS] Arxiv yuklandi! Vaqt: {elapsed:.1f}s, Download URL: {asset_data.get('browser_download_url')}")

    print("\n[VERIFICATION] Yuklab olish havolasi:")
    print(f"https://github.com/{owner}/{repo}/releases/download/{tag}/ComicLab_Portable_v1.0.0.zip")

if __name__ == "__main__":
    main()
