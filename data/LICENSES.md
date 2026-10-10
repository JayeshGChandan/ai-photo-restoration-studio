# Licence record: CLIC 2020 validation sets

Retrieved 2026-10-10. Images live in `data/clic2020/` (gitignored, never
committed); fetch and verify them with `scripts/fetch_data.py` against
`data/manifest.csv`.

## Sources

| Split | Archive (official download host) | Images |
|---|---|---|
| professional | https://downloads.compression.cc/clic2020_professional_valid.zip | 41 PNG |
| mobile | https://downloads.compression.cc/clic2020_mobile_valid.zip | 61 PNG |

Both archives are linked from the CLIC 2020 challenge page,
https://archive.compression.cc/challenge/ ("Validation Dataset P
(“professional”) (129MB)" / "Validation Dataset M (“mobile”) (226MB)"), which
states: "The challenge data is released by the Computer Vision Lab of ETH
Zurich". That page contains no licence statement. Neither archive contains a
licence file (each holds only the PNGs).

## professional — Unsplash License

Source: https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_professional_2020.txt
(Last-Modified 2020-07-17; SHA-256
`d2a4c2e6a98352bd75b03040d1ed6f8f7209f314ff437fe142a3fe0aa4c072a4`).
Linked as "License (Unsplash)" from
https://archive.compression.cc/2021/tasks/index.html, which describes the same
archive as the "professional" set of CLIC2020.

Verbatim text of that file:

> https://unsplash.com/license
>
> All photos published on Unsplash can be used for free. You can use them for commercial and noncommercial purposes. You do not need to ask permission from or provide credit to the photographer or Unsplash, although it is appreciated when possible.
>
> More precisely, Unsplash grants you an irrevocable, nonexclusive, worldwide copyright license to download, copy, modify, distribute, perform, and use photos from Unsplash for free, including for commercial purposes, without permission from or attributing the photographer or Unsplash. This license does not include the right to compile photos from Unsplash to replicate a similar or competing service.

## mobile — CC0 1.0 Universal

Source: https://data.vision.ee.ethz.ch/cvl/clic/LICENSE_mobile_2020.txt
(Last-Modified 2020-07-17; SHA-256
`437ca31a127c9f1b191ff9a42bb563e5f7e414ae819006dca4635d9fb34aeb94`).
The file is the full Creative Commons "CC0 1.0 Universal" legal code (7,171
bytes), beginning verbatim:

> Creative Commons Legal Code
>
> CC0 1.0 Universal

It is served from the same official ETH CVL host and named in parallel with the
professional licence file, but no archive page found links to it directly.
Canonical text: https://creativecommons.org/publicdomain/zero/1.0/legalcode

Clause 4(c), verbatim from that file:

> c. Affirmer disclaims responsibility for clearing rights of other persons
>    that may apply to the Work or any use thereof, including without
>    limitation any person's Copyright and Related Rights in the Work.
>    Further, Affirmer disclaims responsibility for obtaining any necessary
>    consents, permissions or other rights required for any use of the
>    Work.
