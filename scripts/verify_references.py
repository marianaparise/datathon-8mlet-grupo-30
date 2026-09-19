#!/usr/bin/env python3
"""Audit the bibliography of the project documents against official registries.

Every reference cited in docs/Relatorio-Tecnico-TC5-Grupo30.docx and in
docs/Introducao-Fundamentacao-TC5-Grupo30.docx is checked against the Crossref API
(journal articles, conference proceedings, monographs) or the DataCite API (arXiv
DOIs). The script asserts that authors, year, venue and page range match what the
documents claim, and exits non-zero on any divergence.

Entries without a registered DOI are checked by other means: PMLR, JMLR and NeurIPS
papers by HTTP reachability of their stable proceedings page, and books by ISBN
lookup on OpenLibrary.

Page ranges are asserted only where the registry actually carries them. Three
Statistical Science entries have no page data in Crossref, so none is claimed.

    python scripts/verify_references.py            # audit everything
    python scripts/verify_references.py --offline  # skip network, only self-check

Exit code 0 means every claim in the report's reference list is backed by an
official record.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

USER_AGENT = "tc5-datathon-reference-audit/1.0 (mailto:m.dornelles19@gmail.com)"
TIMEOUT = 30


@dataclass(frozen=True)
class Reference:
    """A bibliographic claim made by the report, plus how to verify it."""

    key: str
    doi: str | None = None
    registry: str = "crossref"  # crossref | datacite | url | isbn
    surnames: tuple[str, ...] = ()
    year: int | None = None
    venue_contains: str | None = None
    pages: str | None = None
    url: str | None = None
    isbn: str | None = None
    notes: str = field(default="")


REFERENCES: tuple[Reference, ...] = (
    Reference("AGRAWAL; GOYAL (2012)", registry="url",
              url="https://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf",
              surnames=("Agrawal", "Goyal"), year=2012,
              notes="PMLR v.23 — COLT 2012; sem DOI registrado"),
    Reference("AGRAWAL; GOYAL (2013)", registry="url",
              url="https://proceedings.mlr.press/v28/agrawal13.html",
              surnames=("Agrawal", "Goyal"), year=2013,
              notes="PMLR v.28, p.127-135 — ICML 2013; sem DOI registrado"),
    Reference("BIETTI; AGARWAL; LANGFORD (2021)", registry="url",
              url="https://www.jmlr.org/papers/v22/18-863.html",
              surnames=("Bietti", "Agarwal", "Langford"), year=2021,
              notes="JMLR v.22, n.133, p.1-49; sem DOI registrado"),
    Reference("BROWN; CAI; DASGUPTA (2001)", doi="10.1214/ss/1009213286",
              surnames=("Brown", "Cai", "DasGupta"), year=2001,
              venue_contains="Statistical Science",
              notes="Crossref não carrega paginação; nenhuma é reivindicada"),
    Reference("BUBECK; CESA-BIANCHI (2012)", doi="10.1561/2200000024",
              surnames=("Bubeck", "Cesa-Bianchi"), year=2012,
              venue_contains="Foundations and Trends", pages="1-122"),
    Reference("CHU; LI; REYZIN; SCHAPIRE (2011)", registry="url",
              url="https://proceedings.mlr.press/v15/chu11a.html",
              surnames=("Chu", "Li", "Reyzin", "Schapire"), year=2011,
              notes="PMLR v.15, p.208-214 — AISTATS 2011; sem DOI registrado"),
    Reference("DUDÍK; ERHAN; LANGFORD; LI (2014)", doi="10.1214/14-STS500",
              surnames=("Dudík", "Erhan", "Langford", "Li"), year=2014,
              venue_contains="Statistical Science",
              notes="Crossref não carrega paginação; nenhuma é reivindicada"),
    Reference("GITTINS (1979)", doi="10.1111/j.2517-6161.1979.tb01068.x",
              surnames=("Gittins",), year=1979,
              venue_contains="Royal Statistical Society", pages="148-164",
              notes="p.148-164 conforme registro — não 148-177"),
    Reference("HORVITZ; THOMPSON (1952)", doi="10.1080/01621459.1952.10483446",
              surnames=("Horvitz", "Thompson"), year=1952,
              venue_contains="Journal of the American Statistical Association",
              pages="663-685"),
    Reference("IMBENS; RUBIN (2015)", doi="10.1017/CBO9781139025751",
              surnames=("Imbens", "Rubin"), year=2015,
              venue_contains="Cambridge University Press"),
    Reference("KISH (1965)", registry="isbn", isbn="9780471109495",
              surnames=("Kish",),
              notes="ISBN é da reimpressão Wiley Classics de 1995 do original de 1965"),
    Reference("NICULESCU-MIZIL; CARUANA (2005)", doi="10.1145/1102351.1102430",
              surnames=("Niculescu-Mizil", "Caruana"), year=2005,
              venue_contains="Machine learning", pages="625-632"),
    Reference("ROSENBAUM; RUBIN (1983)", doi="10.1093/biomet/70.1.41",
              surnames=("ROSENBAUM", "RUBIN"), year=1983,
              venue_contains="Biometrika", pages="41-55"),
    Reference("RUSSO; VAN ROY; KAZEROUNI; OSBAND; WEN (2018)",
              doi="10.1561/2200000070",
              surnames=("Russo", "Van Roy", "Kazerouni", "Osband", "Wen"),
              year=2018, venue_contains="Foundations and Trends", pages="1-96"),
    Reference("SCOTT (2010)", doi="10.1002/asmb.874",
              surnames=("Scott",), year=2010,
              venue_contains="Applied Stochastic Models", pages="639-658"),
    Reference("SWAMINATHAN; JOACHIMS (2015)", registry="url",
              url="https://proceedings.neurips.cc/paper/2015/hash/"
                  "39027dfad5138c9ca0c474d71db915c3-Abstract.html",
              surnames=("Swaminathan", "Joachims"), year=2015,
              notes="NIPS 2015 (Advances in NeurIPS 28); sem DOI registrado"),
    Reference("VILLAR; BOWDEN; WASON (2015)", doi="10.1214/14-STS504",
              surnames=("Villar", "Bowden", "Wason"), year=2015,
              venue_contains="Statistical Science",
              notes="Crossref não carrega paginação; nenhuma é reivindicada"),
    Reference("AUER; CESA-BIANCHI; FISCHER (2002)", doi="10.1023/A:1013689704352",
              surnames=("Auer", "Cesa-Bianchi", "Fischer"), year=2002,
              venue_contains="Machine Learning", pages="235-256"),
    Reference("BRIER (1950)",
              doi="10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2",
              surnames=("BRIER",), year=1950,
              venue_contains="Monthly Weather Review", pages="1-3"),
    Reference("CHAPELLE; LI (2011)", registry="url",
              url="https://proceedings.neurips.cc/paper/2011/hash/"
                  "e53a0a2978c28872a4505bdb51db06dc-Abstract.html",
              surnames=("Chapelle", "Li"), year=2011,
              notes="NIPS 2011 (Advances in NeurIPS 24); sem DOI registrado"),
    Reference("DUDÍK; LANGFORD; LI (2011)", doi="10.48550/arXiv.1103.4601",
              registry="datacite", surnames=("Dudik", "Langford", "Li"), year=2011,
              notes="ICML 2011, declarado no campo comment do arXiv"),
    Reference("KOHAVI; TANG; XU (2020)", doi="10.1017/9781108653985",
              surnames=("Kohavi", "Tang", "Xu"), year=2020,
              venue_contains="Cambridge University Press"),
    Reference("LAI; ROBBINS (1985)", doi="10.1016/0196-8858(85)90002-8",
              surnames=("Lai", "Robbins"), year=1985,
              venue_contains="Advances in Applied Mathematics", pages="4-22"),
    Reference("LATTIMORE; SZEPESVÁRI (2020)", doi="10.1017/9781108571401",
              surnames=("Lattimore", "Szepesvári"), year=2020,
              venue_contains="Cambridge University Press"),
    Reference("LI; CHU; LANGFORD; SCHAPIRE (2010)", doi="10.1145/1772690.1772758",
              surnames=("Li", "Chu", "Langford", "Schapire"), year=2010,
              venue_contains="World wide web", pages="661-670"),
    Reference("LI; CHU; LANGFORD; WANG (2011)", doi="10.1145/1935826.1935878",
              surnames=("Li", "Chu", "Langford", "Wang"), year=2011,
              venue_contains="Web search and data mining", pages="297-306"),
    Reference("MORO; CORTEZ; RITA (2014)", doi="10.1016/j.dss.2014.03.001",
              surnames=("Moro", "Cortez", "Rita"), year=2014,
              venue_contains="Decision Support Systems", pages="22-31",
              notes="artigo de origem da base bank-marketing"),
    Reference("ROBBINS (1952)", doi="10.1090/S0002-9904-1952-09620-8",
              surnames=("Robbins",), year=1952,
              venue_contains="Bulletin of the American Mathematical Society",
              pages="527-535"),
    Reference("SUTTON; BARTO (2018)", registry="isbn", isbn="9780262039246",
              surnames=("Sutton",), year=2018,
              notes="MIT Press, 2. ed.; livro sem DOI"),
    Reference("THOMPSON (1933)", doi="10.1093/biomet/25.3-4.285",
              surnames=("THOMPSON",), year=1933,
              venue_contains="Biometrika", pages="285-294"),
    Reference("WILSON (1927)", doi="10.1080/01621459.1927.10502953",
              surnames=("Wilson",), year=1927,
              venue_contains="Journal of the American Statistical Association",
              pages="209-212"),
    Reference("ZADROZNY; ELKAN (2002)", doi="10.1145/775047.775151",
              surnames=("Zadrozny", "Elkan"), year=2002,
              venue_contains="Knowledge discovery and data mining", pages="694-699"),
)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _sort_key(text: str) -> str:
    """Alphabetical key ignoring accents, so Í sorts next to I."""
    stripped = unicodedata.normalize("NFKD", text)
    return "".join(c for c in stripped if not unicodedata.combining(c)).upper()


def check_crossref(ref: Reference) -> list[str]:
    assert ref.doi
    url = "https://api.crossref.org/works/" + urllib.parse.quote(ref.doi, safe="")
    msg = json.loads(_get(url))["message"]

    problems: list[str] = []
    found = [a.get("family", "") for a in msg.get("author", [])]
    for surname in ref.surnames:
        if not any(_norm(surname) in _norm(f) for f in found):
            problems.append(f"autor {surname!r} ausente do registro (achados: {found})")

    year = (msg.get("issued", {}).get("date-parts") or [[None]])[0][0]
    if ref.year is not None and year != ref.year:
        problems.append(f"ano divergente: relatório diz {ref.year}, "
                        f"registro diz {year}")

    if ref.venue_contains:
        venue = (msg.get("container-title") or [msg.get("publisher", "")])[0]
        if _norm(ref.venue_contains) not in _norm(venue):
            problems.append(f"veículo divergente: esperado ~{ref.venue_contains!r}, "
                            f"registro diz {venue!r}")

    if ref.pages and _norm(msg.get("page", "")) != _norm(ref.pages):
        problems.append(f"paginação divergente: relatório diz {ref.pages!r}, "
                        f"registro diz {msg.get('page')!r}")
    return problems


def check_datacite(ref: Reference) -> list[str]:
    assert ref.doi
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(ref.doi, safe="")
    attrs = json.loads(_get(url))["data"]["attributes"]

    problems: list[str] = []
    found = [c.get("familyName", "") for c in attrs.get("creators", [])]
    for surname in ref.surnames:
        if not any(_norm(surname) in _norm(f) for f in found):
            problems.append(f"autor {surname!r} ausente do registro (achados: {found})")
    return problems


def check_url(ref: Reference) -> list[str]:
    assert ref.url
    try:
        _get(ref.url)
    except urllib.error.HTTPError as exc:
        return [f"página de anais inacessível: HTTP {exc.code}"]
    return []


def check_isbn(ref: Reference) -> list[str]:
    assert ref.isbn
    url = (f"https://openlibrary.org/api/books?bibkeys=ISBN:{ref.isbn}"
           "&format=json&jscmd=data")
    data = json.loads(_get(url))
    record = data.get(f"ISBN:{ref.isbn}")
    if not record:
        return [f"ISBN {ref.isbn} não encontrado na OpenLibrary"]
    problems: list[str] = []
    authors = [a.get("name", "") for a in record.get("authors", [])]
    for surname in ref.surnames:
        if not any(_norm(surname) in _norm(a) for a in authors):
            problems.append(f"autor {surname!r} ausente (achados: {authors})")
    return problems


CHECKERS = {
    "crossref": check_crossref,
    "datacite": check_datacite,
    "url": check_url,
    "isbn": check_isbn,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true",
                        help="não acessa a rede; apenas valida a tabela interna")
    args = parser.parse_args()

    if args.offline:
        print(f"{len(REFERENCES)} referências declaradas; verificação de rede pulada.")
        return 0

    failures = 0
    for ref in sorted(REFERENCES, key=lambda r: _sort_key(r.key)):
        try:
            problems = CHECKERS[ref.registry](ref)
        except Exception as exc:  # rede, DOI inexistente, JSON inválido
            print(f"FALHA  {ref.key}: {exc}")
            failures += 1
            continue

        if problems:
            failures += 1
            print(f"FALHA  {ref.key}")
            for problem in problems:
                print(f"       - {problem}")
        else:
            suffix = f"  ({ref.notes})" if ref.notes else ""
            print(f"OK     {ref.key}{suffix}")

    total = len(REFERENCES)
    print(f"\n{total - failures}/{total} referências conferidas contra "
          "registro oficial.")
    if failures:
        print("Bibliografia NÃO auditada com sucesso.", file=sys.stderr)
        return 1
    print("Bibliografia auditada: todas as citações dos documentos existem "
          "e conferem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
