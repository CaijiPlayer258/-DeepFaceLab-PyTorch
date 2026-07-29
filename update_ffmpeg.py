"""
FFmpeg 更新工具
下载 Gyan.dev 完整版 FFmpeg（含 libplacebo、zscale 等 HDR 色调映射支持）
"""
import os, sys, zipfile, shutil, urllib.request, tempfile, subprocess
from pathlib import Path

FFMPEG_DIR = Path(__file__).parent / "ffmpeg"
BACKUP_DIR = FFMPEG_DIR.parent / "ffmpeg_backup"
# 完整版 FFmpeg（shared = 带 DLL，兼容性好）
URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-full-shared.7z"

def get_7z_path() -> Path:
    """查找 7z 可执行文件"""
    candidates = [
        Path(os.environ.get('PROGRAMFILES', r'C:\Program Files')) / '7-Zip' / '7z.exe',
        Path(os.environ.get('PROGRAMW6432', r'C:\Program Files')) / '7-Zip' / '7z.exe',
        Path(os.environ.get('PROGRAMFILES(X86)', r'C:\Program Files (x86)')) / '7-Zip' / '7z.exe',
        Path(r'C:\Program Files\7-Zip\7z.exe'),
        Path(r'C:\Program Files (x86)\7-Zip\7z.exe'),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def download(url: str, dest: Path) -> bool:
    """下载文件并显示进度"""
    print(f"正在下载 FFmpeg 完整版...")
    print(f"  源: {url}")
    print(f"  目标: {dest}")
    print(f"  文件约 100MB，请耐心等待...\n")

    def report(block, blocksize, totalsize):
        done = block * blocksize
        pct = min(100, done * 100 // totalsize) if totalsize > 0 else 0
        sys.stdout.write(f"\r  下载进度: {pct}% ({done // 1024 // 1024}MB / {totalsize // 1024 // 1024}MB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, str(dest), report)
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("  请手动下载并解压:")
        print(f"  1. 打开: {URL}")
        print(f"  2. 解压到: {FFMPEG_DIR}")
        return False
    print(f"\n✅ 下载完成: {dest}")
    return True

def extract_7z(archive: Path, dest: Path) -> bool:
    """解压 7z 文件"""
    sz = get_7z_path()
    if sz is None:
        print("❌ 未找到 7-Zip，请手动解压:")
        print(f"  1. 解压 {archive} 的内容到 {dest}")
        print(f"  2. 确保 {dest / 'ffmpeg.exe'} 存在")
        return False

    print(f"正在解压 (使用 {sz})...")
    # 7z x archive.7z -ooutput_dir -y
    r = subprocess.run([str(sz), "x", str(archive), f"-o{str(dest)}", "-y"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"❌ 解压失败 (exit={r.returncode})")
        print(r.stderr[:500])
        return False
    print("✅ 解压完成")
    return True

def find_exe_in_extracted(dest: Path) -> Path:
    """在解压目录中找到 ffmpeg.exe"""
    for root, dirs, files in os.walk(dest):
        for f in files:
            if f.lower() == 'ffmpeg.exe':
                return Path(root) / f
    return None

def main():
    print("=" * 60)
    print("  FFmpeg 更新工具 — 完整版（HDR 色调映射支持）")
    print("=" * 60)

    # 检查当前版本
    current = FFMPEG_DIR / "ffmpeg.exe"
    if current.exists():
        r = subprocess.run([str(current), "-version"], capture_output=True, text=True)
        ver_line = r.stdout.splitlines()[0] if r.stdout else "?"
        print(f"\n当前版本: {ver_line}")
        if 'full' in r.stdout.lower() or 'libplacebo' in r.stdout:
            print("  当前已经是完整版 ✅")
            yn = input("\n是否仍要重新下载？(y/N): ").strip().lower()
            if yn != 'y':
                print("已取消")
                return

    # 确认
    print(f"\n将下载完整版 FFmpeg (~100MB) 并更新到:")
    print(f"  {FFMPEG_DIR}")
    print("\n注意: 下载完成后旧版本会备份到:")
    print(f"  {BACKUP_DIR}")
    yn = input("\n是否继续？(Y/n): ").strip().lower()
    if yn == 'n':
        print("已取消")
        return

    # 下载
    tmp = Path(tempfile.gettempdir()) / "ffmpeg-full-shared.7z"
    if not download(URL, tmp):
        return

    # 备份旧版
    if current.exists():
        print(f"\n备份旧版到 {BACKUP_DIR}...")
        if BACKUP_DIR.exists():
            shutil.rmtree(str(BACKUP_DIR))
        shutil.copytree(str(FFMPEG_DIR), str(BACKUP_DIR))
        print("✅ 备份完成")

    # 解压到临时目录
    extract_dir = Path(tempfile.gettempdir()) / "ffmpeg_extract"
    if extract_dir.exists():
        shutil.rmtree(str(extract_dir))
    extract_dir.mkdir(parents=True)

    if not extract_7z(tmp, extract_dir):
        print("\n⚠ 解压失败，已恢复备份")
        return

    # 找到 ffmpeg.exe
    exe_path = find_exe_in_extracted(extract_dir)
    if exe_path is None:
        print("❌ 解压后未找到 ffmpeg.exe")
        print("请手动复制到 {FFMPEG_DIR}")
        return

    # 复制文件到 ffmpeg/ 目录
    src_dir = exe_path.parent
    print(f"\n复制文件到 {FFMPEG_DIR}...")

    # 清空目标目录
    for f in FFMPEG_DIR.iterdir():
        if f.is_file():
            f.unlink()
        elif f.is_dir():
            shutil.rmtree(str(f))

    # 复制所有文件
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(FFMPEG_DIR / f.name))
        elif f.is_dir():
            shutil.copytree(str(f), str(FFMPEG_DIR / f.name), dirs_exist_ok=True)

    print("✅ 更新完成！")

    # 验证
    r = subprocess.run([str(FFMPEG_DIR / "ffmpeg.exe"), "-version"], capture_output=True, text=True)
    print(f"\n新版本: {r.stdout.splitlines()[0]}")

    # 检查关键功能
    has_libplacebo = 'libplacebo' in r.stdout
    has_zscale = 'libzimg' in r.stdout
    print(f"  libplacebo (HDR色调映射): {'✅' if has_libplacebo else '❌'}")
    print(f"  zscale (色彩空间转换):   {'✅' if has_zscale else '❌'}")
    print(f"  hevc_nvenc (GPU编码):   {'✅' if '--enable-nvenc' in r.stdout else '❌'}")

    # 清理临时文件
    try:
        tmp.unlink()
        shutil.rmtree(str(extract_dir))
    except:
        pass

    if not has_libplacebo:
        print("\n⚠ 下载的版本仍不支持 libplacebo。")
        print("  如需 HDR 色调映射，请手动安装 FFmpeg full-shared 版本。")

if __name__ == '__main__':
    main()
