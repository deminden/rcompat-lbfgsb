#!/usr/bin/env bash
# Build an external black-box oracle. Never reads optimizer implementation source.
set -euo pipefail

task_prefix=${1:?Usage: build_reference_r462_native.sh EXISTING_CONDA_PREFIX [BUILD_CACHE] [COPIED_FROM_PREFIX]}
task_build_root=${2:-"${HOME}/.cache/rcompat-reference/r462-native"}
task_copied_from=${3:-}
task_scripts=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_version=4.6.2
task_source_sha=c297040f6392b1dac6e5c0eb398422cfdaedf540ae2205edb9ad0d10cb10ee00
mkdir -p "$task_build_root" "$task_build_root/logs"
task_archive="$task_build_root/R-${task_version}.tar.gz"
if [ ! -f "$task_archive" ]; then
    curl --fail --location --retry 2 "https://cran.r-project.org/src/base/R-4/R-${task_version}.tar.gz" --output "$task_archive"
fi
printf '%s  %s\n' "$task_source_sha" "$task_archive" | shasum -a 256 --check
if [ ! -d "$task_build_root/R-${task_version}" ]; then
    tar -xzf "$task_archive" -C "$task_build_root"
fi
mkdir -p "$task_build_root/build"
export PATH="$task_prefix/bin:$PATH"
export CC="$task_prefix/bin/arm64-apple-darwin20.0.0-clang"
export CXX="$task_prefix/bin/arm64-apple-darwin20.0.0-clang++"
export FC="$task_prefix/bin/arm64-apple-darwin20.0.0-gfortran"
export CFLAGS="-O3 -mcpu=native -fPIC"
export CXXFLAGS="-O3 -mcpu=native -fPIC"
export FFLAGS="-O3 -mcpu=native -fPIC"
export FCFLAGS="-O3 -mcpu=native -fPIC"
export CPPFLAGS="-I$task_prefix/include"
export LDFLAGS="-L$task_prefix/lib -Wl,-rpath,$task_prefix/lib"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export R_ENVIRON_USER=/dev/null
export R_PROFILE_USER=/dev/null
cd "$task_build_root/build"
../R-${task_version}/configure --prefix="$task_prefix" --enable-R-shlib \
    --with-blas=-lblas --with-lapack --with-x=no --with-aqua=no \
    --without-tcltk --disable-java > "$task_build_root/logs/configure.log" 2>&1
make -j6 > "$task_build_root/logs/make.log" 2>&1
make install > "$task_build_root/logs/install.log" 2>&1
if [ -n "$task_copied_from" ]; then
    python3 "$task_scripts/repair_reference_links.py" "$task_copied_from" "$task_prefix"
fi
export R_MAKEVARS_USER=/dev/null
export R_MAKEVARS_SITE=/dev/null
"$task_prefix/bin/Rscript" --vanilla "$task_scripts/install_reference_deseq2_native.R" > "$task_build_root/logs/deseq2-install.log" 2>&1
"$task_prefix/bin/Rscript" --vanilla -e '
stopifnot(as.character(getRversion())=="4.6.2")
cat(R.version.string,"\nDESeq2 ",as.character(packageVersion("DESeq2")),"\n",sep="")
for(key in c("CC","CFLAGS","CXX","CXXFLAGS","FC","FFLAGS")) {
 cat(key,": ",system2(file.path(R.home("bin"),"R"),c("CMD","config",key),stdout=TRUE),"\n",sep="")
}
print(sessionInfo())
' | tee "$task_build_root/logs/reference-session.txt"
