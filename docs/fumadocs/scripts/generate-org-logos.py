#!/usr/bin/env python3
"""Generate the publishing-institution logo map for the docs site.

Every model and benchmark identity mark on the docs site shows the logo of the
*publishing institution* (company, industrial lab, university, or research
institute), never a personal GitHub avatar.

The ORGS registry and the MODEL_ORGS / BENCHMARK_ORGS mappings below are
hand-reviewed. Institutions were resolved from the paper author affiliations,
official project pages, and the owning organization (not personal forks) of
`source.repo`. When a paper is a company + university collaboration, the
company logo is used. When several universities co-publish, the first
(or corresponding) affiliation wins.

Logo assets live in `public/org-logos/`. They come from Wikimedia Commons
(official logos with free licenses) and the Simple Icons project (CC0), plus a
few GitHub *organization* brand avatars that were already committed to this
repository for companies without a freely licensed logo. Personal GitHub
avatars are banned: this script must never map an entry to
`avatars.githubusercontent.com` content or derive a logo from a repository
owner's user avatar.

Outputs:
- lib/model-logo-map.json          (orgs + modelLogos + benchmarkLogos)
- lib/benchmark-catalog-status.json (rewrites each `logoKey` to an org id)
- public/org-logos/SOURCES.md       (per-file provenance / license manifest)

Run from `docs/fumadocs/`:
    python3 scripts/generate-org-logos.py
"""

from __future__ import annotations

import json
from pathlib import Path

DOCS_ROOT = Path(__file__).resolve().parents[1]
LOGO_DIR = DOCS_ROOT / "public" / "org-logos"
MAP_PATH = DOCS_ROOT / "lib" / "model-logo-map.json"
BENCH_STATUS_PATH = DOCS_ROOT / "lib" / "benchmark-catalog-status.json"
SOURCES_PATH = LOGO_DIR / "SOURCES.md"

SIMPLE_ICONS = "Simple Icons (https://simpleicons.org), CC0-1.0"


def commons(title: str) -> str:
    return "https://commons.wikimedia.org/wiki/File:" + title.replace(" ", "_")


# --------------------------------------------------------------------------
# Institution registry. `abbr` is shown when no logo file is available.
# `logo=None` means the identity mark renders the abbreviation.
# --------------------------------------------------------------------------
ORGS: dict[str, dict] = {
    # -- Companies / industrial labs -------------------------------------
    "tencent": {
        "name": "Tencent",
        "abbr": "TC",
        "logo": "tencent.svg",
        "source": commons("Tencent Logo.svg"),
        "license": "Public domain (text logo)",
    },
    "alibaba": {
        "name": "Alibaba",
        "abbr": "AL",
        "logo": "alibaba.svg",
        "source": commons("Alibaba en logo.svg"),
        "license": "Public domain (text logo)",
    },
    "amap": {
        "name": "Amap (Alibaba)",
        "abbr": "AM",
        "logo": "alibaba.svg",
        "source": commons("Alibaba en logo.svg"),
        "license": "Public domain (text logo)",
    },
    "nvidia": {
        "name": "NVIDIA",
        "abbr": "NV",
        "logo": "nvidia.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "bytedance": {
        "name": "ByteDance",
        "abbr": "BD",
        "logo": "bytedance.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "meta": {
        "name": "Meta",
        "abbr": "M",
        "logo": "meta.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "google": {
        "name": "Google",
        "abbr": "G",
        "logo": "google.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "google-deepmind": {
        "name": "Google DeepMind",
        "abbr": "GDM",
        "logo": "google-deepmind.svg",
        "source": commons("Google DeepMind logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "openai": {
        "name": "OpenAI",
        "abbr": "OA",
        "logo": "openai.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "microsoft": {
        "name": "Microsoft",
        "abbr": "MS",
        "logo": "microsoft.svg",
        "source": commons("Microsoft icon.svg"),
        "license": "Public domain (simple geometry)",
    },
    "meituan": {
        "name": "Meituan",
        "abbr": "MT",
        "logo": "meituan.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "kuaishou": {
        "name": "Kuaishou",
        "abbr": "KS",
        "logo": "kuaishou.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "huggingface": {
        "name": "Hugging Face",
        "abbr": "HF",
        "logo": "huggingface.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "physical-intelligence": {
        "name": "Physical Intelligence",
        "abbr": "PI",
        "logo": "physical-intelligence.png",
        "source": "GitHub organization brand avatar (github.com/Physical-Intelligence), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "naver": {
        "name": "NAVER",
        "abbr": "NA",
        "logo": "naver.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "sony": {
        "name": "Sony AI",
        "abbr": "SO",
        "logo": "sony.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "adobe": {
        "name": "Adobe",
        "abbr": "AD",
        "logo": "adobe.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "snap": {
        "name": "Snap Inc.",
        "abbr": "SN",
        "logo": "snap.svg",
        "source": SIMPLE_ICONS + " (Snapchat mark)",
        "license": "CC0-1.0",
    },
    "oppo": {
        "name": "OPPO Research Institute",
        "abbr": "OP",
        "logo": "oppo.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "vivo": {
        "name": "vivo",
        "abbr": "VV",
        "logo": "vivo.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "dji": {
        "name": "DJI",
        "abbr": "DJ",
        "logo": "dji.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "minimax": {
        "name": "MiniMax",
        "abbr": "MM",
        "logo": "minimax.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "zhipu": {
        "name": "Zhipu AI",
        "abbr": "ZP",
        "logo": "zhipu.png",
        "source": commons("Zhipu AI.png"),
        "license": "Public domain (text logo)",
    },
    "sensetime": {
        "name": "SenseTime",
        "abbr": "ST",
        "logo": "sensetime.png",
        "source": commons("SenseTime logo.png"),
        "license": "CC BY-SA 4.0",
    },
    "jd": {
        "name": "JD.com (Joy Future Academy)",
        "abbr": "JD",
        "logo": None,
    },
    "antgroup": {
        "name": "Ant Group (Robbyant)",
        "abbr": "AG",
        "logo": "antgroup.png",
        "source": "GitHub organization brand avatar (github.com/robbyant), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "agibot": {
        "name": "AgiBot",
        "abbr": "AB",
        "logo": "agibot.jpg",
        "source": "GitHub organization brand avatar (github.com/AgibotTech), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "skywork": {
        "name": "Skywork AI (Kunlun)",
        "abbr": "SK",
        "logo": "skywork.jpg",
        "source": "GitHub organization brand avatar (github.com/SkyworkAI), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "stepfun": {
        "name": "StepFun",
        "abbr": "SF",
        "logo": "stepfun.png",
        "source": "GitHub organization brand avatar (github.com/stepfun-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "shengshu": {
        "name": "Shengshu Technology",
        "abbr": "SS",
        "logo": "shengshu.jpg",
        "source": "GitHub organization brand avatar (github.com/shengshu-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "lightricks": {
        "name": "Lightricks",
        "abbr": "LT",
        "logo": "lightricks.jpg",
        "source": "GitHub organization brand avatar (github.com/Lightricks), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "luma": {
        "name": "Luma AI",
        "abbr": "LU",
        "logo": "luma.png",
        "source": "GitHub organization brand avatar (github.com/lumalabs-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "genmo": {
        "name": "Genmo",
        "abbr": "GM",
        "logo": "genmo.png",
        "source": "GitHub organization brand avatar (github.com/genmoai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "krea": {
        "name": "Krea AI",
        "abbr": "KR",
        "logo": "krea.png",
        "source": "GitHub organization brand avatar (github.com/krea-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "runway": {
        "name": "Runway",
        "abbr": "RW",
        "logo": "runway.png",
        "source": "GitHub organization brand avatar (github.com/runwayml), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "stability": {
        "name": "Stability AI",
        "abbr": "SA",
        "logo": "stability.png",
        "source": "GitHub organization brand avatar (github.com/Stability-AI), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "sandai": {
        "name": "Sand AI",
        "abbr": "SD",
        "logo": "sandai.jpg",
        "source": "GitHub organization brand avatar (github.com/SandAI-org), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "worldlabs": {
        "name": "World Labs",
        "abbr": "WL",
        "logo": "worldlabs.png",
        "source": "Organization brand avatar, pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "gigaai": {
        "name": "GigaAI",
        "abbr": "GA",
        "logo": "gigaai.png",
        "source": "GitHub organization brand avatar (github.com/open-gigaai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "hpcaitech": {
        "name": "HPC-AI Tech",
        "abbr": "HA",
        "logo": "hpcaitech.jpg",
        "source": "GitHub organization brand avatar (github.com/hpcaitech), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "insta360": {
        "name": "Insta360",
        "abbr": "I3",
        "logo": "insta360.png",
        "source": "GitHub organization brand avatar (github.com/Insta360-Research-Team), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "dexmal": {
        "name": "Dexmal",
        "abbr": "DX",
        "logo": "dexmal.jpg",
        "source": "GitHub organization brand avatar (github.com/Dexmal), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "etched": {
        "name": "Etched",
        "abbr": "ET",
        "logo": "etched.png",
        "source": "GitHub organization brand avatar (github.com/etched-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "rhymes": {
        "name": "Rhymes AI",
        "abbr": "RA",
        "logo": "rhymes.png",
        "source": "GitHub organization brand avatar (github.com/rhymes-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "beingbeyond": {
        "name": "BeingBeyond",
        "abbr": "BB",
        "logo": "beingbeyond.png",
        "source": "GitHub organization brand avatar (github.com/BeingBeyond), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "x-square": {
        "name": "X Square Robot",
        "abbr": "XS",
        "logo": "x-square.png",
        "source": "GitHub organization brand avatar (github.com/X-Square-Robot), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "x-humanoid": {
        "name": "X-Humanoid (Beijing Humanoid Robot Innovation Center)",
        "abbr": "XH",
        "logo": None,
    },
    "spirit-ai": {
        "name": "Spirit AI",
        "abbr": "SP",
        "logo": None,
    },
    "starkware": {
        "name": "StarkWare",
        "abbr": "SW",
        "logo": None,
    },
    "shanda": {
        "name": "Shanda AI",
        "abbr": "SH",
        "logo": None,
    },
    "exla": {
        "name": "Exla AI",
        "abbr": "EX",
        "logo": "exla.png",
        "source": "GitHub organization brand avatar (github.com/exla-ai), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "createai": {
        "name": "CreateAI",
        "abbr": "CR",
        "logo": "createai.png",
        "source": "GitHub organization brand avatar (github.com/IamCreateAI), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "inspatio": {
        "name": "InSpatio",
        "abbr": "IN",
        "logo": "inspatio.jpg",
        "source": "GitHub organization brand avatar (github.com/inspatio), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "meigen": {
        "name": "MeiGen-AI",
        "abbr": "MG",
        "logo": "meigen.jpg",
        "source": "GitHub organization brand avatar (github.com/MeiGen-AI), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "happyoyster": {
        "name": "HappyOyster",
        "abbr": "HO",
        "logo": None,
    },
    "xiaomi": {
        "name": "Xiaomi Robotics",
        "abbr": "MI",
        "logo": "xiaomi.svg",
        "source": SIMPLE_ICONS,
        "license": "CC0-1.0",
    },
    "midea": {
        "name": "Midea Group",
        "abbr": "MD",
        "logo": None,
    },
    "galaxea": {
        "name": "Galaxea AI",
        "abbr": "GX",
        "logo": None,
    },
    "riemann-dynamics": {
        "name": "Riemann Dynamics",
        "abbr": "RD",
        "logo": None,
    },
    # -- Universities ------------------------------------------------------
    "stanford": {
        "name": "Stanford University",
        "abbr": "SU",
        "logo": "stanford.svg",
        "source": commons("Stanford Cardinal logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "berkeley": {
        "name": "UC Berkeley",
        "abbr": "UCB",
        "logo": "berkeley.svg",
        "source": commons("Seal of University of California, Berkeley.svg"),
        "license": "Public domain",
    },
    "mit": {
        "name": "MIT",
        "abbr": "MIT",
        "logo": "mit.svg",
        "source": commons("MIT logo.svg"),
        "license": "Public domain (text logo)",
    },
    "tsinghua": {
        "name": "Tsinghua University",
        "abbr": "THU",
        "logo": "tsinghua.png",
        "source": commons("Tsinghua University Logo.svg"),
        "license": "Public domain",
    },
    "pku": {
        "name": "Peking University",
        "abbr": "PKU",
        "logo": "pku.svg",
        "source": commons("Peking University seal.svg"),
        "license": "Public domain",
    },
    "ustc": {
        "name": "University of Science and Technology of China",
        "abbr": "USTC",
        "logo": None,
    },
    "sjtu": {
        "name": "Shanghai Jiao Tong University",
        "abbr": "SJTU",
        "logo": None,
    },
    "fudan": {
        "name": "Fudan University",
        "abbr": "FDU",
        "logo": "fudan.svg",
        "source": commons("Fudan University Logo.svg"),
        "license": "Public domain",
    },
    "zju": {
        "name": "Zhejiang University",
        "abbr": "ZJU",
        "logo": None,
    },
    "hust": {
        "name": "Huazhong University of Science and Technology",
        "abbr": "HUST",
        "logo": None,
    },
    "buaa": {
        "name": "Beihang University",
        "abbr": "BUAA",
        "logo": None,
    },
    "csu": {
        "name": "Central South University",
        "abbr": "CSU",
        "logo": None,
    },
    "cas": {
        "name": "Chinese Academy of Sciences",
        "abbr": "CAS",
        "logo": None,
    },
    "ucas": {
        "name": "University of Chinese Academy of Sciences",
        "abbr": "UCAS",
        "logo": None,
    },
    "hku": {
        "name": "The University of Hong Kong",
        "abbr": "HKU",
        "logo": None,
    },
    "cuhk": {
        "name": "The Chinese University of Hong Kong",
        "abbr": "CUHK",
        "logo": None,
    },
    "cityu": {
        "name": "City University of Hong Kong",
        "abbr": "CityU",
        "logo": None,
    },
    "hkust": {
        "name": "Hong Kong University of Science and Technology",
        "abbr": "HKUST",
        "logo": None,
    },
    "ntu": {
        "name": "Nanyang Technological University",
        "abbr": "NTU",
        "logo": None,
    },
    "nus": {
        "name": "National University of Singapore",
        "abbr": "NUS",
        "logo": None,
    },
    "kaist": {
        "name": "KAIST",
        "abbr": "KAIST",
        "logo": "kaist.svg",
        "source": commons("KAIST logo.svg"),
        "license": "Public domain",
    },
    "snu": {
        "name": "Seoul National University",
        "abbr": "SNU",
        "logo": "snu.svg",
        "source": commons("서울대학교.svg"),
        "license": "Public domain (simple shield rendition)",
    },
    "oxford": {
        "name": "University of Oxford",
        "abbr": "OX",
        "logo": "oxford.svg",
        "source": commons("Oxford-University-Circlet.svg"),
        "license": "Public domain",
    },
    "imperial": {
        "name": "Imperial College London",
        "abbr": "ICL",
        "logo": "imperial.png",
        "source": commons("Shield of Imperial College London.svg"),
        "license": "CC BY 3.0",
    },
    "eth": {
        "name": "ETH Zurich",
        "abbr": "ETH",
        "logo": "eth.svg",
        "source": commons("ETH Zürich Logo black.svg"),
        "license": "Public domain (text logo)",
    },
    "unige": {
        "name": "University of Geneva",
        "abbr": "UNIGE",
        "logo": "unige.svg",
        "source": commons("Uni GE logo.svg"),
        "license": "Public domain",
    },
    "freiburg": {
        "name": "University of Freiburg",
        "abbr": "UFR",
        "logo": "freiburg.png",
        "source": commons("Wortmarke-grundform-universitaet freiburg blau rgb.png"),
        "license": "CC BY-SA 4.0",
    },
    "upc": {
        "name": "Universitat Politècnica de Catalunya (IRI, CSIC-UPC)",
        "abbr": "UPC",
        "logo": "upc.svg",
        "source": commons("Logo UPC.svg"),
        "license": "Public domain",
    },
    "mila": {
        "name": "Mila – Quebec AI Institute",
        "abbr": "MILA",
        "logo": "mila.svg",
        "source": commons("Mila logo.svg"),
        "license": "Public domain (text logo)",
    },
    "epfl": {
        "name": "EPFL",
        "abbr": "EPFL",
        "logo": "epfl.svg",
        "source": commons("Logo EPFL.svg"),
        "license": "Public domain (text logo)",
    },
    "nyu": {
        "name": "New York University",
        "abbr": "NYU",
        "logo": "nyu.svg",
        "source": commons("New York University Seal.svg"),
        "license": "Public domain",
    },
    "umich": {
        "name": "University of Michigan",
        "abbr": "UM",
        "logo": "umich.svg",
        "source": commons("University of Michigan logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "usc": {
        "name": "University of Southern California",
        "abbr": "USC",
        "logo": "usc.svg",
        "source": commons("USC Trojans logo.svg"),
        "license": "Public domain (text logo)",
    },
    "ucsd": {
        "name": "UC San Diego",
        "abbr": "UCSD",
        "logo": "ucsd.svg",
        "source": commons("Seal of the University of California, San Diego.svg"),
        "license": "Public domain",
    },
    "utaustin": {
        "name": "The University of Texas at Austin",
        "abbr": "UT",
        "logo": "utaustin.svg",
        "source": commons("Texas Longhorns logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "umass": {
        "name": "UMass Amherst",
        "abbr": "UMA",
        "logo": "umass.svg",
        "source": commons("UMass Amherst athletics logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "uwash": {
        "name": "University of Washington",
        "abbr": "UW",
        "logo": "uwash.svg",
        "source": commons("Washington Huskies logo.svg"),
        "license": "Public domain (text logo)",
    },
    "waterloo": {
        "name": "University of Waterloo",
        "abbr": "UWL",
        "logo": None,
    },
    "jhu": {
        "name": "Johns Hopkins University",
        "abbr": "JHU",
        "logo": "jhu.png",
        "source": commons("Johns Hopkins University logo.png"),
        "license": "Public domain (text logo)",
    },
    "cmu": {
        "name": "Carnegie Mellon University",
        "abbr": "CMU",
        "logo": "cmu.svg",
        "source": commons("Carnegie Mellon wordmark.svg"),
        "license": "Public domain (text logo)",
    },
    "sfu": {
        "name": "Simon Fraser University",
        "abbr": "SFU",
        "logo": "sfu.png",
        "source": commons("SFU logo.png"),
        "license": "Public domain (text logo)",
    },
    "adelaide": {
        "name": "The University of Adelaide (AIML)",
        "abbr": "UA",
        "logo": "adelaide.png",
        "source": commons("Arms of the University of Adelaide.svg"),
        "license": "CC BY-SA 4.0",
    },
    "tamu": {
        "name": "Texas A&M University",
        "abbr": "TAMU",
        "logo": "tamu.svg",
        "source": commons("Texas A&M University logo.svg"),
        "license": "Public domain (text logo)",
    },
    "gatech": {
        "name": "Georgia Institute of Technology",
        "abbr": "GT",
        "logo": "gatech.svg",
        "source": commons("Georgia Tech Yellow Jackets logo.svg"),
        "license": "Public domain (text logo)",
    },
    "caltech": {
        "name": "California Institute of Technology (Caltech)",
        "abbr": "CIT",
        "logo": None,
    },
    "koc": {
        "name": "Koç University",
        "abbr": "KU",
        "logo": "koc.svg",
        "source": commons("Koç University logo.svg"),
        "license": "Public domain",
    },
    "sjsu": {
        "name": "San José State University",
        "abbr": "SJSU",
        "logo": "sjsu.svg",
        "source": commons("San Jose State University logo.svg"),
        "license": "Public domain",
    },
    # -- Institutes / foundations -----------------------------------------
    "ai2": {
        "name": "Allen Institute for AI (Ai2)",
        "abbr": "AI2",
        "logo": "ai2.svg",
        "source": commons("Allen Institute for Artificial Intelligence.svg"),
        "license": "Public domain (text logo)",
    },
    "baai": {
        "name": "Beijing Academy of Artificial Intelligence",
        "abbr": "BAAI",
        "logo": "baai.svg",
        "source": commons("Beijing Academy of Artificial Intelligence logo.svg"),
        "license": "Public domain (simple geometry)",
    },
    "shanghai-ai-lab": {
        "name": "Shanghai AI Laboratory",
        "abbr": "SAIL",
        "logo": None,
    },
    "airi": {
        "name": "AIRI (Artificial Intelligence Research Institute)",
        "abbr": "AIRI",
        "logo": None,
    },
    "farama": {
        "name": "Farama Foundation",
        "abbr": "FR",
        "logo": "farama.png",
        "source": "GitHub organization brand avatar (github.com/Farama-Foundation), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "benchcouncil": {
        "name": "BenchCouncil",
        "abbr": "BC",
        "logo": "benchcouncil.jpg",
        "source": "GitHub organization brand avatar (github.com/BenchCouncil), pre-existing repo asset",
        "license": "Organization brand mark",
    },
    "cau": {
        "name": "CAU HAI Lab",
        "abbr": "CAU",
        "logo": None,
    },
    "nu-lab": {
        "name": "NU World Model & Embodied AI Lab",
        "abbr": "NU",
        "logo": None,
    },
    "worldfoundry": {
        "name": "WorldFoundry",
        "abbr": "WF",
        "logo": None,
    },
}

# --------------------------------------------------------------------------
# model_id -> org id (hand-reviewed; see module docstring for the rules).
# Models intentionally left out fall back to model-name initials in the UI
# because no publishing institution could be confirmed.
# --------------------------------------------------------------------------
MODEL_ORGS: dict[str, str] = {
    "4d-gs": "hust",
    "abot-m0": "amap",
    "abot-world-0-5b-lf": "amap",
    "ac3d": "snap",
    "act": "stanford",
    "adaworld": "hkust",
    "allegro": "rhymes",
    "animatediff": "cuhk",
    "astra": "tsinghua",
    "ati-wan21-14b": "bytedance",
    "being-h05": "beingbeyond",
    "being-h07": "beingbeyond",
    "bernini": "bytedance",
    "bernini-r-1.3b": "bytedance",
    "bernini-r-14b": "bytedance",
    "cameractrl": "cuhk",
    "causal-forcing": "tsinghua",
    "causal-rcm": "nvidia",
    "cogact": "microsoft",
    "cogvideox": "zhipu",
    "consistent4d": "ucas",
    "cosmos-predict-2": "nvidia",
    "cosmos-predict-2.5": "nvidia",
    "cosmos-transfer-2.5": "nvidia",
    "cosmos3": "nvidia",
    "ctrl-world": "tsinghua",
    "cut3r": "berkeley",
    "d-nerf": "upc",
    "dap": "insta360",
    "db-cogact": "dexmal",
    "depth-anything-v1": "bytedance",
    "depth-anything-v2-prior": "bytedance",
    "depth-anything-v3": "bytedance",
    "depth-anything-v3-prior": "bytedance",
    "diamond": "unige",
    "diffusion-policy": "stanford",
    "dino-wm": "nyu",
    "dm0": "dexmal",
    "dreamdojo": "nvidia",
    "dreamx-world-5b": "amap",
    "dreamx-world-5b-cam": "amap",
    "dreamzero": "nvidia",
    "droid-w": "eth",
    "dualcamctrl": "hkust",
    "dust3r": "naver",
    "dust3r-base-model": "naver",
    "dvlt": "nvidia",
    "dynamicrafter": "tencent",
    "easyanimate": "alibaba",
    "echo-infinity": "jd",
    "echo-memory-block-ssm": "jd",
    "echo-memory-context-k1": "jd",
    "echo-memory-context-k20": "jd",
    "echo-memory-spatial": "jd",
    "echo-memory-spatial-concat-text": "jd",
    "echo-memory-spatial-cross-attn-t32": "jd",
    "echo-memory-spatial-no-injection": "jd",
    "echo-memory-ssm-ctx1-every4-hint21": "jd",
    "echo-memory-ssm-ctx5-every1-hint21": "jd",
    "echo-memory-ssm-ctx5-every4-hint81": "jd",
    "echo-memory-videossm-hybrid": "jd",
    "egowm": "cmu",
    "emu3.5": "baai",
    "eo1": "shanghai-ai-lab",
    "eventvla": "shanghai-ai-lab",
    "fantasyworld": "amap",
    "flashworld": "tencent",
    "framepack": "stanford",
    "galaxea-vla": "galaxea",
    "gamma-world": "nvidia",
    "gaussian-actor": "huggingface",
    "gaussianflow": "usc",
    "gaussianobject": "sjtu",
    "gen3c": "nvidia",
    "genie-envisioner": "agibot",
    "geocalib-prior": "eth",
    "geometry-prior": "worldfoundry",
    "giga-brain-0": "gigaai",
    "giga-world-0": "gigaai",
    "giga-world-policy-0.5": "gigaai",
    "go1": "agibot",
    "gr00t": "nvidia",
    "h-rdt": "tsinghua",
    "hailuo-2p3": "minimax",
    "happyoyster": "happyoyster",
    "helios": "pku",
    "hexplane": "umich",
    "hma": "mit",
    "hunyuan-game-craft": "tencent",
    "hy-embodied-vla": "tencent",
    "hunyuanvideo": "tencent",
    "hunyuanvideo-1.5": "tencent",
    "hunyuanworld-1": "tencent",
    "hunyuanworld-mirror": "tencent",
    "hunyuanworld-voyager": "tencent",
    "hy-embodied": "tencent",
    "hy-world-2.0": "tencent",
    "hy-worldplay": "tencent",
    "hydra": "hust",
    "hyworld-worldgen": "tencent",
    "i2vgen-xl": "alibaba",
    "infinite-vggt": "berkeley",
    "infinite-world": "meigen",
    "inspatio-world": "inspatio",
    "internvla-a1": "shanghai-ai-lab",
    "irasim": "bytedance",
    "iris": "unige",
    "k-planes": "berkeley",
    "kairos-sensenova": "sensetime",
    "kling-api": "kuaishou",
    "krea-realtime-video": "krea",
    "lagernvs": "meta",
    "lapa": "microsoft",
    "last-r1": "cuhk",
    "leworldmodel": "mila",
    "libero-para": "cau",
    "lingbot-map": "antgroup",
    "lingbot-va": "antgroup",
    "lingbot-video": "antgroup",
    "lingbot-vla": "antgroup",
    "lingbot-vla-v2": "antgroup",
    "lingbot-world": "antgroup",
    "lingbot-world-act": "antgroup",
    "lingbot-world-v2": "antgroup",
    "liveworld": "adelaide",
    "loger": "berkeley",
    "longcat-video": "meituan",
    "longvie-1": "shanghai-ai-lab",
    "longvie-2": "shanghai-ai-lab",
    "ltx-2.x": "lightricks",
    "ltx-video": "lightricks",
    "luma-ray2": "luma",
    "lyra": "nvidia",
    "magi-1": "sandai",
    "magicworld": "vivo",
    "matrix-game-1": "skywork",
    "matrix-game-2": "skywork",
    "matrix-game-3": "skywork",
    "matrix-game-3.5-first-person": "riemann-dynamics",
    "matrix-game-3.5-third-person": "riemann-dynamics",
    "metric3d-prior": "dji",
    "mineworld": "microsoft",
    "minwm-hy-action2v": "shengshu",
    "minwm-wan-action2v": "shengshu",
    "mmaudio": "sony",
    "mochi-1": "genmo",
    "modelscope-t2v": "alibaba",
    "molmoact2": "ai2",
    "molmobot": "ai2",
    "monst3r": "google-deepmind",
    "mosaicmem": "starkware",
    "motionbricks": "nvidia",
    "motionctrl": "tencent",
    "multi-task-dit": "huggingface",
    "mvdiffusion": "sfu",
    "neoverse": "createai",
    "nwm": "meta",
    "oasis-500m": "etched",
    "octo": "berkeley",
    "omnivinci": "nvidia",
    "open-magvit2": "tencent",
    "open-sora": "hpcaitech",
    "open-sora-plan": "pku",
    "openpi": "physical-intelligence",
    "openpie-0.6": "exla",
    "openvla": "stanford",
    "openvla-oft": "stanford",
    "pandora": "ucsd",
    "pi0": "physical-intelligence",
    "pi0-fast": "physical-intelligence",
    "pi0-worldfoundry": "physical-intelligence",
    "pi05": "physical-intelligence",
    "pi3": "sjtu",
    "pixelsplat": "mit",
    "pointworld": "nvidia",
    "prior-depth-anything": "zju",
    "pusa-vidgen": "cityu",
    "qwen2.5-omni": "alibaba",
    "rdt-1b": "tsinghua",
    "real-time-chunking": "physical-intelligence",
    "recammaster": "kuaishou",
    "rise": "shanghai-ai-lab",
    "roboflamingo": "bytedance",
    "rolling-forcing": "tencent",
    "rt-1": "google",
    "runway-gen4p5": "runway",
    "sama-14b": "tsinghua",
    "sana": "nvidia",
    "sana-wm": "nvidia",
    "sc-gs": "hku",
    "scope": "ucas",
    "self-forcing": "adobe",
    "shape-of-motion": "berkeley",
    "shotstream": "kuaishou",
    "show-o": "nus",
    "skyreels-v2": "skywork",
    "skyreels-v3": "skywork",
    "smolvla": "huggingface",
    "solaris": "nyu",
    "sora2": "openai",
    "spatia": "microsoft",
    "spatial-ladder": "zju",
    "spatial-reasoner": "jhu",
    "spirit-v1.5": "spirit-ai",
    "splatt3r": "oxford",
    "stable-video-infinity": "epfl",
    "stable-virtual-camera": "stability",
    "starwm": "cas",
    "step-video-t2v": "stepfun",
    "t2v_turbo_t2v": "google",
    "tdmpc": "ucsd",
    "tesseract": "umass",
    "thinksound": "alibaba",
    "tineuvox": "hust",
    "tinyvla": "midea",
    "track-anything-prior": "zju",
    "uni3c": "alibaba",
    "unianimate-dit": "alibaba",
    "unidepth-v2-prior": "eth",
    "unik3d-prior": "eth",
    "uwm": "uwash",
    "vchitect-2-t2v": "shanghai-ai-lab",
    "veo3": "google-deepmind",
    "versecrafter": "tencent",
    "vggt": "meta",
    "vggt-omega": "meta",
    "vid2world": "tsinghua",
    "video-depth-anything-prior": "bytedance",
    "videocrafter": "tencent",
    "videocrafter1-i2v": "tencent",
    "videocrafter1-t2v": "tencent",
    "videocrafter2-t2v": "tencent",
    "viewcrafter": "tencent",
    "vlanext": "ntu",
    "vmem": "oxford",
    "vqbet": "snu",
    "wall-oss": "x-square",
    "wan-2p5": "alibaba",
    "wan-2p6": "alibaba",
    "wan-2p7": "alibaba",
    "wan2.1": "alibaba",
    "wan2.1-vace": "alibaba",
    "wan2.2": "alibaba",
    "wan21-fun-14b-cam": "alibaba",
    "wan21-fun-1p3b-cam": "alibaba",
    "wan22-fun-5b-cam": "alibaba",
    "wan22-fun-a14b-cam": "alibaba",
    "warp-as-history": "sjtu",
    "wilddet3d": "ai2",
    "wildworld": "shanda",
    "wonderjourney": "stanford",
    "wonderworld": "stanford",
    "worldcam": "kaist",
    "worldfm": "inspatio",
    "worldgen": "meta",
    "worldlabs": "worldlabs",
    "worldlabs-marble-1.1": "worldlabs",
    "worldmem": "ntu",
    "wow": "x-humanoid",
    "xiaomi-robotics-0": "xiaomi",
    "xiaomi-robotics-1": "xiaomi",
    "xvla": "tsinghua",
    "yume": "fudan",
}

# Models whose publishing institution could not be confirmed from the paper,
# project page, or repository organization. They intentionally have no logo
# and render model-name initials (never author names or GitHub avatars).
UNRESOLVED_MODELS: list[str] = [
    "a1",
    "ahawam",
    "alayaworld",
    "dexora-1b",
    "fastwam",
    "lda-1b",
    "mem-0",
    "mira",
    "mme-vla",
    "moverse",
    "multiworld",
    "omniforcing",
    "open-dreamer",
    "simworld",
    "spatial-forcing",
    "starvla",
    "vggt-world",
    "worldgrow",
    "x-wam",
    "zeroscope",
]

# --------------------------------------------------------------------------
# benchmark_id -> org id (hand-reviewed).
# --------------------------------------------------------------------------
BENCHMARK_ORGS: dict[str, str] = {
    "4dworldbench": "ustc",
    "ai2thor": "ai2",
    "aigcbench": "benchcouncil",
    "apple-pi": "ntu",
    "behavior1k": "stanford",
    "bridgedata-v2": "berkeley",
    "calvin": "freiburg",
    "camerabench": "cmu",
    "chronomagic-bench": "pku",
    "devil-dynamics": "cas",
    "evalcrafter": "tencent",
    "ewmbench": "agibot",
    "genai-bench": "waterloo",
    "ipv-bench": "nus",
    "iworld-bench": "tsinghua",
    "kinetix": "oxford",
    "larybench": "meituan",
    "libero": "utaustin",
    "libero-mem": "utaustin",
    "libero-para": "cau",
    "libero-plus": "utaustin",
    "libero-pro": "utaustin",
    "likephys": "oxford",
    "maniskill": "ucsd",
    "maniskill2": "ucsd",
    "metaworld": "farama",
    "mikasa": "airi",
    "mind": "csu",
    "molmospaces": "ai2",
    "phyfps-bench-gen": "tamu",
    "phygenbench": "shanghai-ai-lab",
    "phyground": "nu-lab",
    "physical-ai-bench": "gatech",
    "physics-iq": "google-deepmind",
    "physics-iq-verified": "google-deepmind",
    "physvidbench": "koc",
    "rbench": "pku",
    "rlbench": "imperial",
    "robocasa": "nvidia",
    "robocerebra": "buaa",
    "robotwin": "hku",
    "sana-wm-bench": "nvidia",
    "simpler-env": "google-deepmind",
    "stevo-bench": "caltech",
    "t2v-compbench": "hku",
    "t2v-safety-bench": "cas",
    "t2vworldbench": "sjsu",
    "vbench": "shanghai-ai-lab",
    "vbench-2.0": "shanghai-ai-lab",
    "vbench-plus-plus": "shanghai-ai-lab",
    "videophy": "google",
    "videophy2": "google",
    "videoscience-bench": "ucsd",
    "videoscore": "waterloo",
    "videoverse": "oppo",
    "visual-chronometer": "tamu",
    "vlabench": "fudan",
    "vmbench": "amap",
    "wbench": "meituan",
    "world-in-world": "jhu",
    "worldarena": "tsinghua",
    "worldmodelbench": "nvidia",
    "worldolympiad": "alibaba",
    "worldscore": "stanford",
    "wrbench": "ustc",
}

# Benchmarks whose publishing institution could not be confirmed; they render
# benchmark-name initials.
UNRESOLVED_BENCHMARKS: list[str] = [
    "fetv",
    "memobench",
    "mirabench",
    "phyeduvideo",
    "robomme",
    "video-bench",
    "worldbench",
    "worldreasonbench",
]


def build_org_entries() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    for org_id, meta in sorted(ORGS.items()):
        entry: dict = {
            "key": org_id,
            "name": meta["name"],
            "abbr": meta["abbr"],
        }
        logo = meta.get("logo")
        if logo:
            path = LOGO_DIR / logo
            if not path.is_file():
                raise SystemExit(f"missing logo asset for {org_id}: {path}")
            entry["src"] = f"/org-logos/{logo}"
        if meta.get("source"):
            entry["source"] = meta["source"]
        if meta.get("license"):
            entry["license"] = meta["license"]
        entries[org_id] = entry
    return entries


def validate() -> None:
    for table_name, table in (("MODEL_ORGS", MODEL_ORGS), ("BENCHMARK_ORGS", BENCHMARK_ORGS)):
        for item_id, org_id in table.items():
            if org_id not in ORGS:
                raise SystemExit(f"{table_name}[{item_id!r}] -> unknown org {org_id!r}")
    overlap_m = set(MODEL_ORGS) & set(UNRESOLVED_MODELS)
    overlap_b = set(BENCHMARK_ORGS) & set(UNRESOLVED_BENCHMARKS)
    if overlap_m or overlap_b:
        raise SystemExit(f"entries both mapped and unresolved: {overlap_m | overlap_b}")


def write_map(orgs: dict[str, dict]) -> None:
    payload = {
        "generatedBy": "docs/fumadocs/scripts/generate-org-logos.py (hand-reviewed publishing-institution mapping; GitHub user avatars are banned)",
        "orgs": orgs,
        "modelLogos": dict(sorted(MODEL_ORGS.items())),
        "benchmarkLogos": dict(sorted(BENCHMARK_ORGS.items())),
        "unmapped": {
            "models": sorted(UNRESOLVED_MODELS),
            "benchmarks": sorted(UNRESOLVED_BENCHMARKS),
        },
    }
    MAP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"wrote {MAP_PATH}")


def rewrite_benchmark_status() -> None:
    if not BENCH_STATUS_PATH.is_file():
        print(f"skip (missing): {BENCH_STATUS_PATH}")
        return
    data = json.loads(BENCH_STATUS_PATH.read_text())
    changed = 0
    for bench_id, record in data.items():
        if "logoKey" not in record:
            continue
        new_key = BENCHMARK_ORGS.get(bench_id, "")
        if record["logoKey"] != new_key:
            record["logoKey"] = new_key
            changed += 1
    BENCH_STATUS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(f"rewrote {changed} logoKey values in {BENCH_STATUS_PATH}")


def write_sources(orgs: dict[str, dict]) -> None:
    used_by: dict[str, list[str]] = {}
    for model_id, org_id in MODEL_ORGS.items():
        used_by.setdefault(org_id, []).append(f"model:{model_id}")
    for bench_id, org_id in BENCHMARK_ORGS.items():
        used_by.setdefault(org_id, []).append(f"benchmark:{bench_id}")

    lines = [
        "# Organization logo sources",
        "",
        "Generated by `docs/fumadocs/scripts/generate-org-logos.py`.",
        "Every asset below is the mark of a publishing institution.",
        "No file in this directory is a personal GitHub avatar.",
        "",
        "| File | Institution | Source | License | Used by |",
        "| --- | --- | --- | --- | --- |",
    ]
    for org_id, entry in sorted(orgs.items()):
        if "src" not in entry:
            continue
        fname = entry["src"].split("/")[-1]
        source = entry.get("source", "-")
        license_ = entry.get("license", "-")
        usage = ", ".join(sorted(used_by.get(org_id, []))) or "-"
        lines.append(f"| `{fname}` | {entry['name']} | {source} | {license_} | {usage} |")
    lines.append("")
    lines.append("Institutions without a committed logo render their abbreviation instead:")
    abbr_only = [
        f"`{org_id}` ({entry['name']}, \"{entry['abbr']}\")"
        for org_id, entry in sorted(orgs.items())
        if "src" not in entry
    ]
    lines.append(", ".join(abbr_only) + ".")
    lines.append("")
    SOURCES_PATH.write_text("\n".join(lines))
    print(f"wrote {SOURCES_PATH}")


def main() -> None:
    validate()
    orgs = build_org_entries()
    write_map(orgs)
    rewrite_benchmark_status()
    write_sources(orgs)
    total_models = len(MODEL_ORGS) + len(UNRESOLVED_MODELS)
    total_benches = len(BENCHMARK_ORGS) + len(UNRESOLVED_BENCHMARKS)
    with_logo = {k for k, v in ORGS.items() if v.get("logo")}
    print(
        f"models: {len(MODEL_ORGS)}/{total_models} mapped, "
        f"benchmarks: {len(BENCHMARK_ORGS)}/{total_benches} mapped, "
        f"orgs: {len(ORGS)} ({len(with_logo)} with logo files)"
    )


if __name__ == "__main__":
    main()
