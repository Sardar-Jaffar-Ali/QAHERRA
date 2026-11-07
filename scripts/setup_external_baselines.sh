#!/usr/bin/env bash
# Download the exact upstream revisions used to define the controlled adapters.
set -euo pipefail

mkdir -p external

if [[ ! -d external/Telco-RAG/.git ]]; then
  git clone https://github.com/netop-team/Telco-RAG.git external/Telco-RAG
fi
git -C external/Telco-RAG fetch --all
git -C external/Telco-RAG checkout df92a34d3ad824fe4ef45b81fc956498a353a943

if [[ ! -d external/Chat3GPP/.git ]]; then
  git clone https://github.com/huangl22/Chat3GPP.git external/Chat3GPP
fi
git -C external/Chat3GPP fetch --all
git -C external/Chat3GPP checkout ce65e04da7474ff972516dd807b0278c1bfbf3b3

echo "External baseline revisions are ready under external/."
