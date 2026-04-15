#!/usr/bin/env bash

# apt search, apt download, and dpkg -x are used to obtain MUMPS library paths without installing system-wide.

# Installer steps:
# 1) Check required system tools.
# 2) Create a temporary workspace.
# 3) Download and unpack `libmumps-dev`.
# 4) Detect matching MUMPS version and library path.
# 5) Download matching MUMPS source headers.
# 6) Export `MUMPS_INC`, `MUMPS_LIB`, and `MUMPS_SOLVERS`.
# 7) Install `mumps4py` from local repo (or clone fallback).

# Stop on errors, unset variables, and failed pipes.
set -euo pipefail

# Directory where this script lives.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify required system tools are available.
for command in apt dpkg wget tar find python3 git; do
	if ! command -v "$command" >/dev/null 2>&1; then
		echo "Missing required command: $command" >&2
		exit 1
	fi
done

# Create an isolated temporary working directory and remove it on exit.
workdir="$(mktemp -d -p "${PWD}" mumps-installer-XXXXXX)"
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

# Download the Debian package that contains MUMPS development files.
apt download libmumps-dev

# Detect the downloaded package filename.
deb_file="$(ls libmumps-dev_*.deb | head -n 1)"
if [[ -z "${deb_file}" ]]; then
	echo "Could not locate downloaded libmumps-dev package." >&2
	exit 1
fi

# Extract the MUMPS upstream version from the Debian package version.
version="$(echo "$deb_file" | cut -d'_' -f2 | cut -d'-' -f1)"

# Unpack the package payload to locate libraries.
dpkg -x "$deb_file" dist

# Find the MUMPS library directory inside extracted files.
lib_file="$(find dist -type f \( -name 'libdmumps.*' -o -name 'libmumps_common.*' \) | head -n 1)"
if [[ -z "${lib_file}" ]]; then
	echo "Could not locate MUMPS libraries in extracted package." >&2
	exit 1
fi
lib_folder="$(dirname "$lib_file")"

# Download matching MUMPS sources to get headers.
archive="MUMPS_${version}.tar.gz"
wget "https://mumps-solver.org/${archive}"
tar -xf "$archive"

includes="MUMPS_${version}/include"

# Export environment variables consumed by mumps4py build.
export MUMPS_INC="${workdir}/${includes}"
export MUMPS_LIB="${workdir}/${lib_folder}"
export MUMPS_SOLVERS="dmumps,cmumps,zmumps,smumps"

echo "MUMPS_INC=${MUMPS_INC}"
echo "MUMPS_LIB=${MUMPS_LIB}"
echo "MUMPS_SOLVERS=${MUMPS_SOLVERS}"

# Use local repository when available; otherwise clone from GitHub.
if [[ -d "${script_dir}/../.git" ]]; then
	repo_path="${script_dir}/.."
else
	repo_path="${workdir}/mumps4py"
	git clone https://github.com/imadki/mumps4py.git "$repo_path"
fi

# Install the Python package with current MUMPS environment settings.
python3 -m pip install "$repo_path"
