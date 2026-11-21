"""
Build paper/refs.bib from Crossref (every DOI is resolved; nothing is typed by hand
except the three venues below that have no DOI).

Run once with network access:  python paper/make_bib.py
"""

import re
import urllib.parse
import urllib.request
from pathlib import Path

KEYS = {
    # references of the submitted version (all resolved; [5], [9], [15] metadata corrected)
    "dossantos2025landscape": "10.1109/ACCESS.2025.3596871",
    "lee2025incremental": "10.1109/ACCESS.2025.3547976",
    "wang2025nugnn": "10.1007/s42514-025-00224-3",
    "song2026capgnn": "10.1016/j.neucom.2026.132978",
    "mukhopadhyay2024sparsity": "10.1145/3673038.3673152",
    "zhang2026nm": "10.1016/j.jpdc.2026.105264",
    "sahebi2025hashgrid": "10.1016/j.future.2024.107497",
    "wei2025biobjective": "10.1016/j.neunet.2024.106823",
    "yang2026anchoring": "10.1016/j.neunet.2025.108411",
    "hevapathige2026permutation": "10.1016/j.neunet.2026.108869",
    "cao2025ddqn": "10.1038/s41598-025-16768-x",
    "eyubov2025freight": "10.1007/s00453-024-01291-8",
    "ragno2026coclustering": "10.1007/s10618-026-01211-0",
    "antonietti2025magnet": "10.1007/s00366-025-02223-y",
    "bruglieri2025mingap": "10.1016/j.cor.2025.107224",
    # added in the revision
    "karypis1998metis": "10.1137/S1064827595287997",
    "kernighan1970": "10.1002/j.1538-7305.1970.tb01770.x",
    "fiduccia1982": "10.1109/DAC.1982.1585498",
    "stanton2012ldg": "10.1145/2339530.2339722",
    "tsourakakis2014fennel": "10.1145/2556195.2556213",
    "schloegel2001wavefront": "10.1109/71.926167",
    "schloegel2000unified": "10.1109/SC.2000.10035",
    "hendrickson2000models": "10.1016/S0167-8191(00)00048-X",
    "martella2017spinner": "10.1109/ICDE.2017.153",
    "malewicz2010pregel": "10.1145/1807167.1807184",
    "paranjape2017motifs": "10.1145/3018661.3018731",
    "shun2013ligra": "10.1145/2442516.2442530",
    "lam2015numba": "10.1145/2833157.2833162",
    "sanders2013kahip": "10.1007/978-3-642-38527-8_16",
    "sen2008collective": "10.1609/aimag.v29i3.2157",
    "mu2023streaming": "10.1016/j.neunet.2023.06.026",
    "deng2024taser": "10.1109/IPDPS57955.2024.00087",
    "damiand2024volume": "10.1007/s00366-024-02008-9",
    "zhang2026survey": "10.1109/TKDE.2026.3667713",
    "farkas2026softpart": "10.3390/a19100900",
}

MANUAL = r"""
@inproceedings{hamilton2017graphsage,
  author    = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title     = {Inductive Representation Learning on Large Graphs},
  booktitle = {Advances in Neural Information Processing Systems 30 (NeurIPS)},
  pages     = {1024--1034},
  year      = {2017}
}

@inproceedings{fey2019pyg,
  author    = {Fey, Matthias and Lenssen, Jan Eric},
  title     = {Fast Graph Representation Learning with {PyTorch Geometric}},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year      = {2019}
}

@inproceedings{yang2016planetoid,
  author    = {Yang, Zhilin and Cohen, William W. and Salakhutdinov, Ruslan},
  title     = {Revisiting Semi-Supervised Learning with Graph Embeddings},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  series    = {PMLR},
  volume    = {48},
  pages     = {40--48},
  year      = {2016}
}
"""


def fetch(doi):
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}/transform/application/x-bibtex"
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0"})
    return urllib.request.urlopen(req, timeout=30).read().decode("utf-8")


def protect_title(title: str) -> str:
    """Brace tokens whose case BibTeX must keep (acronyms, CamelCase), e.g. CaPGNN -> {CaPGNN}."""
    proper = {"Python", "Pregel", "Ligra", "Numba", "Spinner", "Metis", "Q"}
    out = []
    for tok in title.split(" "):
        core = tok.strip(":,.;()")
        if core and "{" not in tok and (any(c.isupper() for c in core[1:]) or "$" in core
                                        or core in proper):
            tok = tok.replace(core, "{" + core + "}", 1)
        out.append(tok)
    return " ".join(out)


def fix_title_field(bib: str) -> str:
    start = bib.find("title={")
    if start < 0:
        return bib
    i, depth = start + len("title={"), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(bib[i], 0)
        i += 1
    value = bib[start + len("title={"): i - 1]
    return bib[: start + len("title={")] + protect_title(value) + bib[i - 1:]


def main():
    out = []
    for key, doi in KEYS.items():
        bib = fetch(doi).strip()
        bib = re.sub(r"^@(\w+)\{[^,]*,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)
        bib = bib.replace("&amp;", r"\&").replace("–", "--").replace("’", "'").replace("′", "'")
        bib = bib.replace("$$", "$")
        keep_publisher = bib.startswith("@inbook") or bib.startswith("@book")
        drop = "url|ISSN|ISBN|collection|month|series" + ("" if keep_publisher else "|publisher")
        bib = re.sub(r"\s(" + drop + r")=(\{[^{}]*\}|\w+),", "", bib)
        bib = fix_title_field(bib)
        bib = re.sub(r", (\w+)=", r",\n  \1=", bib)  # one field per line
        out.append(bib)
        print("ok", key)
    path = Path(__file__).with_name("refs.bib")
    path.write_text("\n\n".join(out) + "\n" + MANUAL, encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
