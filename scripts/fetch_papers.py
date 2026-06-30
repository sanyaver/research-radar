#!/usr/bin/env python3
"""Fetch and score recent ML papers from arxiv for 3D & robotics ML pipeline.

Papers accumulate over time — new papers are merged into existing data rather
than replacing it, so the library grows indefinitely.
"""

import requests
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import time
import os

# ── Pipeline sections, ordered by current priority (higher = more important) ─

SECTIONS = [
    {
        "id": "3B_articulation",
        "label": "3B · Articulation & Joints",
        "color": "#EF4444",
        "priority": 10,
        "title_kw": [
            "articulation", "articulated", "joint prediction", "kinematic",
            "part decomposition", "SAMPart", "HoloPart", "PartField",
            "Articulate-Anything", "Particulate", "SPARK", "PhysX-Anything",
            "URDF generation", "movable part", "articulate",
        ],
        "abstract_kw": [
            "articulated object", "joint prediction", "part decomposition",
            "URDF", "kinematic chain", "hinge joint", "prismatic joint",
            "ball joint", "revolute joint", "degree of freedom",
            "PartNet-Mobility", "GAPartNet", "AKB-48",
            "movable parts", "articulation prediction", "joint axis",
            "joint type", "joint parameter",
        ],
    },
    {
        "id": "3A_scene_layout",
        "label": "3A · Scene Layout",
        "color": "#3B82F6",
        "priority": 10,
        "title_kw": [
            "scene layout", "room layout", "scene generation", "scene synthesis",
            "indoor scene", "object placement", "scene arrangement",
        ],
        "abstract_kw": [
            "scene layout", "room layout", "furniture placement",
            "scene generation", "object arrangement", "indoor scene",
            "collision-free placement", "reachability", "PhyScene", "ATISS",
            "DiffuScene", "Holodeck", "3D-RE-GEN", "LayoutVLM",
            "scene composition", "spatial arrangement",
        ],
    },
    {
        "id": "sim_ready",
        "label": "Sim-Ready / Robotics",
        "color": "#06B6D4",
        "priority": 9,
        "title_kw": [
            "Isaac Sim", "simulation", "sim-to-real",
            "robot manipulation", "robot learning", "embodied AI",
            "physics simulation", "sim-ready",
        ],
        "abstract_kw": [
            "Isaac Sim", "Isaac Lab", "robot simulator",
            "sim-to-real", "domain randomization", "robot manipulation",
            "OpenUSD", "USD scene", "physics simulation",
            "collision mesh", "VLA", "vision-language-action",
            "synthetic data pipeline", "NVIDIA Cosmos",
            "reality gap", "sim2real",
        ],
    },
    {
        "id": "2_3d_gen",
        "label": "2 · 3D Generators",
        "color": "#10B981",
        "priority": 7,
        "title_kw": [
            "3D generation", "image-to-3D", "text-to-3D",
            "3D reconstruction", "multi-view reconstruction", "single-view 3D",
            "novel view synthesis", "feed-forward reconstruction",
        ],
        "abstract_kw": [
            "3D generation", "image-to-3D", "text-to-3D",
            "3D reconstruction", "multi-view reconstruction",
            "gaussian splatting", "3DGS", "TRELLIS", "TripoSG",
            "Hunyuan3D", "shape generation", "single-image 3D",
            "Seed3D", "feed-forward 3D",
        ],
    },
    {
        "id": "1_foundations",
        "label": "1 · Foundations",
        "color": "#8B5CF6",
        "priority": 5,
        "title_kw": [
            "gaussian splatting", "neural radiance field", "NeRF",
            "flow matching", "2D Gaussian", "VGGT",
        ],
        "abstract_kw": [
            "3D Gaussian splatting", "neural radiance field",
            "flow matching", "score matching", "2DGS",
            "VGGT", "feed-forward reconstruction",
        ],
    },
    {
        "id": "4B_texturing",
        "label": "4B · Texturing & Materials",
        "color": "#F59E0B",
        "priority": 6,
        "title_kw": [
            "texturing", "texture synthesis", "material estimation",
            "PBR", "appearance synthesis", "relighting", "delighting",
            "material map", "MaterialAnything", "LumiTex", "UniTEX",
        ],
        "abstract_kw": [
            "texture synthesis", "texturing", "PBR material",
            "material estimation", "BRDF", "normal map",
            "roughness map", "metallic map", "delighting",
            "relighting", "material map", "MaterialAnything",
            "LumiTex", "UniTEX", "Pixal3D", "albedo",
        ],
    },
    {
        "id": "4A_topology",
        "label": "4A · Clean Topology",
        "color": "#F97316",
        "priority": 5,
        "title_kw": [
            "retopology", "quad mesh", "mesh generation",
            "remeshing", "MeshAnything", "TreeMeshGPT",
        ],
        "abstract_kw": [
            "retopology", "quad mesh", "artist-friendly mesh",
            "mesh simplification", "remeshing", "topology optimization",
            "MeshAnything", "structured mesh", "mesh topology",
        ],
    },
    {
        "id": "4C_uv",
        "label": "4C · UV Unwrapping",
        "color": "#EC4899",
        "priority": 4,
        "title_kw": [
            "UV unwrapping", "UV mapping", "mesh parameterization",
            "seam prediction", "SeamGPT", "texture atlas",
        ],
        "abstract_kw": [
            "UV unwrapping", "UV mapping", "seam placement",
            "mesh parameterization", "texture atlas", "xatlas",
            "atlas generation", "UV seam",
        ],
    },
]

CODE_TERMS = [
    "open source", "open-source", "publicly available", "code available",
    "weights available", "github.com", "huggingface.co", "open weight",
    "released", "open-weights",
]


def score_paper(title: str, abstract: str) -> tuple:
    title_lo = title.lower()
    abstract_lo = abstract.lower()

    section_scores = {}
    for sec in SECTIONS:
        s = 0
        w = sec["priority"]
        for kw in sec["title_kw"]:
            if kw.lower() in title_lo:
                s += w * 5
        for kw in sec["abstract_kw"]:
            if kw.lower() in abstract_lo:
                s += w * 1
        if s > 0:
            section_scores[sec["id"]] = (s, sec)

    total = sum(v[0] for v in section_scores.values())
    score = min(int(total ** 0.65), 100)

    has_code = any(t in abstract_lo or t in title_lo for t in CODE_TERMS)

    matched = sorted(
        [
            {"id": sid, "label": sec["label"], "color": sec["color"], "score": s}
            for sid, (s, sec) in section_scores.items()
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return score, matched[:4], has_code


def fetch_arxiv(days_back: int = 7) -> list:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    categories = ["cs.CV", "cs.GR", "cs.RO", "cs.LG"]
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)

    papers = []
    seen_ids: set = set()

    for start in range(0, 500, 100):
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={cat_query}"
            f"&start={start}&max_results=100"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            resp = requests.get(url, timeout=45)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Warning: request failed at start={start}: {e}")
            break

        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entries = root.findall("a:entry", ns)
        if not entries:
            break

        oldest_in_batch = None
        for entry in entries:
            pub_str = entry.find("a:published", ns).text.strip()
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            if oldest_in_batch is None or pub_dt < oldest_in_batch:
                oldest_in_batch = pub_dt
            if pub_dt < cutoff:
                continue

            url_elem = entry.find("a:id", ns).text.strip()
            paper_id = url_elem.split("/abs/")[-1]
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)

            title = " ".join(entry.find("a:title", ns).text.split())
            abstract = " ".join(entry.find("a:summary", ns).text.split())
            authors = [
                a.find("a:name", ns).text.strip()
                for a in entry.findall("a:author", ns)
            ]
            cats = [c.get("term") for c in entry.findall("a:category", ns)]

            score, sections, has_code = score_paper(title, abstract)
            if score < 8:
                continue

            papers.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract[:700] + ("…" if len(abstract) > 700 else ""),
                "authors": authors[:6],
                "published": pub_str,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "categories": cats[:4],
                "relevance_score": score,
                "sections": sections,
                "has_code": has_code,
            })

        print(f"  Batch start={start}: {len(entries)} entries, {len(papers)} passing so far")

        if oldest_in_batch and oldest_in_batch < cutoff:
            print(f"  Reached {days_back}-day cutoff, stopping.")
            break

        time.sleep(3)  # arxiv rate limit: 3s between requests

    papers.sort(key=lambda p: p["relevance_score"], reverse=True)
    return papers


def load_existing(path: str = "data/papers.json") -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        papers = {p["id"]: p for p in data.get("papers", [])}
        print(f"  Loaded {len(papers)} existing papers from library")
        return papers
    except FileNotFoundError:
        print("  No existing library, starting fresh")
        return {}
    except Exception as e:
        print(f"  Warning: could not read existing library: {e}")
        return {}


def main():
    now_iso = datetime.now(tz=timezone.utc).isoformat()

    print("Loading existing library...")
    existing = load_existing()

    print("Fetching new papers from arxiv...")
    new_papers = fetch_arxiv(days_back=7)
    print(f"  {len(new_papers)} papers passed relevance filter")

    # Tag first_seen: preserve existing timestamp, set now for genuinely new ones
    for p in new_papers:
        if p["id"] in existing:
            p["first_seen"] = existing[p["id"]].get("first_seen", p["published"])
        else:
            p["first_seen"] = now_iso

    # Merge: start from existing library, overwrite with fresh fetched papers
    # (refreshes scores for re-appearing papers while keeping old ones intact)
    merged = dict(existing)
    for p in new_papers:
        merged[p["id"]] = p

    # Ensure all old papers have first_seen set
    for pid, p in merged.items():
        if "first_seen" not in p:
            p["first_seen"] = p.get("published", now_iso)

    all_papers = sorted(merged.values(), key=lambda p: p["relevance_score"], reverse=True)

    new_count = sum(1 for p in new_papers if p["id"] not in existing)
    print(f"Library: {len(all_papers)} total papers ({new_count} new this run)")

    os.makedirs("data", exist_ok=True)
    output = {
        "last_updated": now_iso,
        "total": len(all_papers),
        "new_this_run": new_count,
        "papers": all_papers,
    }
    with open("data/papers.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("Saved to data/papers.json")


if __name__ == "__main__":
    main()
