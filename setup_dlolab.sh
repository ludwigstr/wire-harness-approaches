#!/bin/bash
# DLO-Lab with the per-element stiffness / rest-curvature patch used for rows 15-18.
set -e
cd "$(dirname "$0")"
mkdir -p extern
if [ ! -d extern/DLO-Lab-perjoint ]; then
  git clone https://github.com/UMass-Embodied-AGI/DLO-Lab extern/DLO-Lab-perjoint
fi
cd extern/DLO-Lab-perjoint
git checkout c5026a9416b03c6bc5186eba13cd4ffd4c0e7796
git apply ../../patches/dlo-lab-perjoint.patch
echo "DLO-Lab ready in extern/DLO-Lab-perjoint (install its requirements as in its README)"
