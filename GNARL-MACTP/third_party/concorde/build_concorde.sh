#!/bin/bash
set -e

CONCORDE_VERSION="03.12.19"
CONCORDE_URL="https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz"
QSOPT_URL="https://www.math.uwaterloo.ca/~bico/qsopt/downloads/codes/ubuntu/qsopt"

# Download Concorde if not already present
if [ ! -f co031219.tgz ]; then
  echo "Downloading Concorde TSP Solver..."
  wget "$CONCORDE_URL" -O co031219.tgz
fi

# Download QSopt if not already present
if [ ! -d QSopt ]; then
  echo "Downloading QSopt LP solver..."
  mkdir -p QSopt
  cd QSopt
  wget "$QSOPT_URL".h -O qsopt.h
  wget "$QSOPT_URL".a -O qsopt.a
  cd ..
fi
qsopt_path=$(pwd)/QSopt

# Unzip Concorde if not already extracted
if [ ! -d concorde ]; then
  echo "Extracting Concorde..."
  tar xvf co031219.tgz
fi

cd concorde

# Build Concorde with QSopt support
echo "Building Concorde with QSopt support..."
./configure --with-qsopt=${qsopt_path}
make

echo "Concorde build complete."
echo "Executable is at: $(pwd)/TSP/concorde"
