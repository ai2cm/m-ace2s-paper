# Third-party notices

M-ACE2S paper Copyright © 2026 The Allen Institute for Artificial Intelligence

This software includes code from the following open source projects. Each is listed with its license and required notices.

This repository does not vendor third-party code. The projects below are the direct dependencies declared in `pyproject.toml` (including the optional `private` extra), at the versions pinned there. Their own dependencies are installed by `uv sync` from PyPI and ship with their own license files. Packages offered under a choice of licenses are listed under the license used here; orjson and tqdm, which are each under two licenses at once, are listed under both.

---

## MIT License

- **choreographer** 1.3.0 — https://github.com/plotly/choreographer Copyright © Plotly, Inc.
- **kaleido** 1.3.0 — https://github.com/plotly/kaleido Copyright © Plotly, Inc
- **logistro** 2.0.1 — https://github.com/geopozo/logistro Copyright © 2025 GeoPozo
- **orjson** 3.11.9 — https://github.com/ijl/orjson Copyright © 2018-2026 ijl; 2020-2021 Aviram Hassan; 2021 Ben Sully; 2022 Nazar Kostetskyi; 2021-2022 Eric Jolibois; 2020 Baul; 2020, 2023 Anders Kaseorg; 2023 Jack Amadeo; 2023 Marc Mueller
- **plotly** 6.8.0 — https://github.com/plotly/plotly.py Copyright © 2016-2024 Plotly Technologies Inc.
- **PyYAML** 6.0.3 — https://github.com/yaml/pyyaml Copyright © 2017-2021 Ingy döt Net; 2006-2016 Kirill Simonov
- **tqdm** 4.68.2 — https://github.com/tqdm/tqdm Copyright © 2016 Google Inc.; 2013 Noam Yorav-Raphael; other tqdm contributors (see commit logs)
- **wandb** 0.29.0 — https://github.com/wandb/wandb Copyright © 2021 Weights and Biases, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Notes: orjson is used under MIT (it offers Apache-2.0 OR MIT) and also contains MPL-2.0 code (see below). tqdm is MPL-2.0 AND MIT: Casper da Costa-Luis's contributions (all files) are MPL-2.0 (see below); other contributors' work, including the original code by Noam Yorav-Raphael and Google Inc., is MIT.

---

## BSD 3-Clause ("New" or "Revised") License

- **ipykernel** 7.2.0 — https://github.com/ipython/ipykernel Copyright © 2015, IPython Development Team
- **jupyterlab** 4.5.7 — https://github.com/jupyterlab/jupyterlab Copyright © 2015-2025 Project Jupyter Contributors
- **nbconvert** 7.17.1 — https://github.com/jupyter/nbconvert Copyright © 2001-2015, IPython Development Team; 2015-, Jupyter Development Team
- **nbformat** 5.10.4 — https://github.com/jupyter/nbformat Copyright © 2001-2015, IPython Development Team; 2015-, Jupyter Development Team
- **numpy** 2.4.6 — https://github.com/numpy/numpy Copyright © 2005-2025, NumPy Developers.
- **pandas** 3.0.3 — https://github.com/pandas-dev/pandas Copyright © 2008-2011, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team; 2011-2026, Open source contributors.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Mozilla Public License 2.0

- **orjson** 3.11.9 — https://github.com/ijl/orjson Copyright © 2018-2026 ijl
- **tqdm** 4.68.2 — https://github.com/tqdm/tqdm Copyright © 2015-2026 Casper da Costa-Luis

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
