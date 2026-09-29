"""
scripts/import_profiler.py - Step 1 Import Profiler for Streamlit Cloud.
Measures cgroup file (mmap/page cache), anon (heap), and RSS deltas for each heavy dependency:
torch, torchvision, cv2, skimage, ctranslate2, easyocr.
"""
import os
import sys
import time
import json
import tempfile
from typing import Dict, List, Any, Optional


def get_cgroup_memory_breakdown() -> Dict[str, float]:
    """Reads cgroup v1/v2 file, anon, shmem and process tree RSS."""
    total_rss = 0.0
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        total_rss += proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except Exception:
                pass
    except Exception:
        pass
    rss_mb = total_rss / (1024 * 1024)

    cg2_stat = "/sys/fs/cgroup/memory.stat"
    cg1_stat = "/sys/fs/cgroup/memory/memory.stat"
    
    file_mb = 0.0
    anon_mb = 0.0
    shmem_mb = 0.0
    
    stat_file = None
    if os.path.exists(cg2_stat):
        stat_file = cg2_stat
    elif os.path.exists(cg1_stat):
        stat_file = cg1_stat
        
    if stat_file:
        try:
            stats = {}
            with open(stat_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            stats[parts[0]] = int(parts[1])
                        except ValueError:
                            pass
            if "file" in stats:
                file_mb = stats["file"] / (1024 * 1024)
            elif "total_cache" in stats:
                file_mb = stats["total_cache"] / (1024 * 1024)
            elif "cache" in stats:
                file_mb = stats["cache"] / (1024 * 1024)
                
            if "anon" in stats:
                anon_mb = stats["anon"] / (1024 * 1024)
            elif "total_rss" in stats:
                anon_mb = stats["total_rss"] / (1024 * 1024)
            elif "rss" in stats:
                anon_mb = stats["rss"] / (1024 * 1024)
                
            if "shmem" in stats:
                shmem_mb = stats["shmem"] / (1024 * 1024)
            elif "total_shmem" in stats:
                shmem_mb = stats["total_shmem"] / (1024 * 1024)
        except Exception:
            pass

    return {
        "file_mb": round(file_mb, 1),
        "anon_mb": round(anon_mb, 1),
        "shmem_mb": round(shmem_mb, 1),
        "rss_mb": round(rss_mb, 1),
    }


def get_pkg_disk_size_mb(mod_name: str) -> Optional[float]:
    """Calculates disk size of package directory in site-packages."""
    try:
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, "__file__") and mod.__file__:
            pkg_dir = os.path.dirname(mod.__file__)
            total = 0
            for root, dirs, files in os.walk(pkg_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    if not os.path.islink(fp):
                        total += os.path.getsize(fp)
            return round(total / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def profile_imports(force: bool = False) -> List[Dict[str, Any]]:
    """Profiles imports sequentially and logs cgroup memory deltas."""
    cache_path = os.path.join(tempfile.gettempdir(), "comic_lab_import_profile.json")
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    print("\n" + "=" * 98, flush=True)
    print("[IMPORT_PROFILER] STEP 1: Profiling per-library cgroup memory deltas...", flush=True)
    print("=" * 98, flush=True)

    libraries_to_profile = [
        "torch",
        "torchvision",
        "cv2",
        "skimage",
        "ctranslate2",
        "easyocr",
    ]

    initial = get_cgroup_memory_breakdown()
    prev = dict(initial)
    results = [
        {
            "step": 0,
            "library": "0. Baseline (Streamlit)",
            "file_mb": initial["file_mb"],
            "delta_file_mb": 0.0,
            "anon_mb": initial["anon_mb"],
            "delta_anon_mb": 0.0,
            "rss_mb": initial["rss_mb"],
            "delta_rss_mb": 0.0,
            "disk_size_mb": 0.0,
            "import_time_s": 0.0,
        }
    ]

    import importlib
    for i, lib in enumerate(libraries_to_profile, 1):
        t0 = time.time()
        try:
            importlib.import_module(lib)
            elapsed = round(time.time() - t0, 3)
            curr = get_cgroup_memory_breakdown()
            disk_mb = get_pkg_disk_size_mb(lib) or 0.0
            
            d_file = round(curr["file_mb"] - prev["file_mb"], 1)
            d_anon = round(curr["anon_mb"] - prev["anon_mb"], 1)
            d_rss = round(curr["rss_mb"] - prev["rss_mb"], 1)
            
            entry = {
                "step": i,
                "library": lib,
                "file_mb": curr["file_mb"],
                "delta_file_mb": d_file,
                "anon_mb": curr["anon_mb"],
                "delta_anon_mb": d_anon,
                "rss_mb": curr["rss_mb"],
                "delta_rss_mb": d_rss,
                "disk_size_mb": disk_mb,
                "import_time_s": elapsed,
            }
            results.append(entry)
            prev = dict(curr)
        except Exception as e:
            results.append({
                "step": i,
                "library": f"{lib} (ERROR: {e})",
                "file_mb": prev["file_mb"],
                "delta_file_mb": 0.0,
                "anon_mb": prev["anon_mb"],
                "delta_anon_mb": 0.0,
                "rss_mb": prev["rss_mb"],
                "delta_rss_mb": 0.0,
                "disk_size_mb": 0.0,
                "import_time_s": 0.0,
            })

    # Print breakdown table to stdout
    print(f"{'Library':<24} | {'file (MB)':<10} | {'+file (MB)':<10} | {'anon (MB)':<10} | {'+anon (MB)':<10} | {'Tree RSS':<10} | {'Disk (MB)':<10} | {'Time (s)':<8}", flush=True)
    print("-" * 98, flush=True)
    for r in results:
        df_str = f"+{r['delta_file_mb']:.1f}" if r['delta_file_mb'] > 0 else f"{r['delta_file_mb']:.1f}"
        da_str = f"+{r['delta_anon_mb']:.1f}" if r['delta_anon_mb'] > 0 else f"{r['delta_anon_mb']:.1f}"
        if r['step'] == 0:
            df_str = "-"
            da_str = "-"
        print(
            f"{r['library']:<24} | {r['file_mb']:<10.1f} | {df_str:<10} | {r['anon_mb']:<10.1f} | {da_str:<10} | {r['rss_mb']:<10.1f} | {r['disk_size_mb']:<10.1f} | {r['import_time_s']:<8.3f}",
            flush=True
        )
    print("=" * 98 + "\n", flush=True)

    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return results


def get_cached_profile() -> Optional[List[Dict[str, Any]]]:
    """Retrieves cached profile results if available."""
    cache_path = os.path.join(tempfile.gettempdir(), "comic_lab_import_profile.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None
