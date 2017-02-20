mapd2
=====

Central repo for the buildout of MapD V2

# Building MapD

MapD uses CMake for its build system. Only the `Unix Makefiles` and `Ninja` generators are regularly used. Others, such as `Xcode` might need some work.

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j

The following `cmake`/`ccmake` options can enable/disable different features:

- `-DENABLE_CALCITE=on` enable Calcite SQL parser. Default `on`.
- `-DENABLE_RENDERING=on` enable backend rendering. Default `off`.
- `-DENABLE_CUDA=off` disable CUDA. Default `on`.
- `-DMAPD_IMMERSE_DOWNLOAD=on` download the latest master build of Immerse / `mapd2-frontend`. Default `on`.
- `-DMAPD_DOCS_DOWNLOAD=on` download the latest master build of the documentation / `docs.mapd.com`. Default `off`. Note: this is a >50MB download.
- `-DPREFER_STATIC_LIBS=on` static link dependencies, if available. Default `off`.

# Dependencies

MapD has the following dependencies:

- [CMake 3.3+](https://cmake.org/)
- [LLVM 3.8](http://llvm.org/)
- [GCC 4.9+](http://gcc.gnu.org/): not required if building with Clang
- [Boost 1.5.7+](http://www.boost.org/)
- [Thrift 0.9.2+](https://thrift.apache.org/)
- [bison++](https://code.google.com/p/flexpp-bisonpp/)
- [Google glog](https://github.com/google/glog)
- [Go 1.5+](https://golang.org/)
- [OpenJDK](http://openjdk.java.net/)
- [CUDA 7.0+](http://nvidia.com/cuda)
- [GLEW](http://glew.sourceforge.net/)
- [GLFW 3.1.2+](http://www.glfw.org/)
- [libpng](http://libpng.org/pub/png/libpng.html)
- [libcurl](https://curl.haxx.se/)
- [crypto++](https://www.cryptopp.com/)
- [gperftools](https://github.com/gperftools/gperftools)
- [gdal](http://gdal.org/)

Generating the documentation requires ((`pip` && `virtualenv`) || `sphinx`), and `texlive` (specifically `pdflatex`). `pip` and `virtualenv` will attempt to install `sphinx` if it is not available.

Dependencies for `mapd_web_server` and other Go utils are in [`ThirdParty/go`](ThirdParty/go). See [`ThirdParty/go/src/mapd/vendor/README.md`](ThirdParty/go/src/mapd/vendor/README.md) for instructions on how to add new deps.

## CentOS 6/7

[scripts/mapd-deps-linux.sh](scripts/mapd-deps-linux.sh) is provided that will automatically download, build, and install most dependencies. Before running this script, make sure you have the basic build tools installed:

    yum groupinstall -y "Development Tools"
    yum install -y zlib-devel \
                   libssh \
                   openssl-devel \
                   openldap-devel \
                   ncurses-devel \
                   git \
                   maven \
                   java-1.8.0-openjdk{-devel,-headless} \
                   gperftools{,-devel,-libs} \
                   libX11-devel \
                   libXv \
                   mesa-libGL-devel \
                   mesa-libGLU \
                   xorg-x11-server-Xorg

For generating the documentation you will also need:

    yum install -y python-pip python-virtualenv
    yum install -y texlive texlive-latex-bin-bin "texlive-*"
    pip install sphinx==1.4.9

Instructions for installing CUDA are below.

### CUDA

CUDA should be installed via the .rpm method, following the instructions provided by Nvidia. Make sure you first enable EPEL via `yum install epel-release`.

### Environment Variables

[scripts/mapd-deps-linux.sh](scripts/mapd-deps-linux.sh) generates two files with the appropriate environment variables: `mapd-deps-<date>.sh` (for sourcing from your shell config) and `mapd-deps-<date>.modulefile` (for use with [Environment Modules](http://modules.sf.net), yum package `environment-modules`). These files are placed in mapd-deps install directory, usually `/usr/local/mapd-deps/<date>`. Either of these may be used to configure your environment: the `.sh` may be sourced in your shell config; the `.modulefile` needs to be moved to the modulespath.

The Java server lib directory containing `libjvm.so` must also be added to your `LD_LIBRARY_PATH`. Add one of the following to your shell config:

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/jre/lib/amd64/server
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/jvm/java-1.8.0/jre/lib/amd64/server

## Mac OS X

[scripts/mapd-deps-osx.sh](scripts/mapd-deps-osx.sh) is provided that will automatically install and/or update [Homebrew](http://brew.sh/) and use that to install all dependencies. Please make sure OS X is completely update to date and Xcode is installed before running. Xcode can be installed from the App Store.

The last command in `mapd-deps-osx.sh` installs MacTeX via Homebrew. Since this is a fairly large package, downloading and installing it may take some time. You can alternatively install directly from their website at [http://www.tug.org/mactex/mactex-download.html](http://www.tug.org/mactex/mactex-download.html).

### CUDA

`mapd-deps-osx.sh` will automatically install CUDA via Homebrew and add the correct environment variables to `~/.bash_profile`.

### Java

`mapd-deps-osx.sh` will automatically install Java and Maven via Homebrew and add the correct environment variables to `~/.bash_profile`.

## Ubuntu 16.04, 16.10

Note: as of 2016-10-17 CUDA 8 does not officially support GCC 6, which is the default in Ubuntu 16.10. For the time being it is recommended that you stick with Ubuntu 16.04 if your require CUDA support.

Most build dependencies are available via APT. Thrift is the one exception and must be built by hand (Thrift 0.9.1 is available in APT, but that version is not supported by MapD).

    apt-get update
    apt-get install build-essential \
                    cmake \
                    cmake-curses-gui \
                    clang-3.8 \
                    clang-format-3.8 \
                    llvm-3.8 \
                    llvm-3.8-dev \
                    libboost-all-dev \
                    libgoogle-glog-dev \
                    golang \
                    libssl-dev \
                    libevent-dev \
                    libglew-dev \
                    libglfw3-dev \
                    libpng-dev \
                    libcurl4-openssl-dev \
                    libcrypto++-dev \
                    xserver-xorg \
                    libglu1-mesa \
                    default-jre \
                    default-jre-headless \
                    default-jdk \
                    default-jdk-headless \
                    maven \
                    libldap2-dev \
                    libncurses5-dev \
                    libglewmx-dev \
                    google-perftools \
                    libgoogle-perftools-dev \
                    libgdal-dev

    apt-get build-dep thrift-compiler
    wget http://apache.claz.org/thrift/0.9.3/thrift-0.9.3.tar.gz
    tar xvf thrift-0.9.3.tar.gz
    cd thrift-0.9.3
    patch -p1 < /path/to/mapd2/scripts/mapd-deps-thrift-refill-buffer.patch
    ./configure --with-lua=no --with-python=no --with-php=no --with-ruby=no --prefix=/usr/local/mapd-deps
    make -j $(nproc)
    make install
    apt-get install bison++

Next you need to configure symlinks so that `clang`, etc point to the newly installed `clang-3.8`:

    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/lib/llvm-3.8/bin/llvm-config 1
    update-alternatives --install /usr/bin/llc llc /usr/lib/llvm-3.8/bin/llc 1
    update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-3.8/bin/clang 1
    update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-3.8/bin/clang++ 1
    update-alternatives --install /usr/bin/clang-format clang-format /usr/lib/llvm-3.8/bin/clang-format 1

For generating the documentation you will also need:

    apt-get install python-pip virtualenv
    apt-get install texlive-latex-base texlive-full
    pip install sphinx==1.4.9

### CUDA

CUDA should be installed via the .deb method, following the instructions provided by Nvidia.

### Environment Variables

CUDA, Java, and mapd-deps need to be added to `LD_LIBRARY_PATH`; CUDA and mapd-deps also need to be added to `PATH`. The easiest way to do so is by creating a new file `/etc/profile.d/mapd-deps.sh` containing the following:

    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/lib/jvm/default-java/jre/lib/amd64/server:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/local/mapd-deps/lib:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=/usr/local/mapd-deps/lib64:$LD_LIBRARY_PATH

    PATH=/usr/local/cuda/bin:$PATH
    PATH=/usr/local/mapd-deps/bin:$PATH

    export LD_LIBRARY_PATH PATH

## Arch

Assuming you already have [yaourt](https://wiki.archlinux.org/index.php/Yaourt) or some other manager that supports the AUR installed:

    yaourt -S git cmake boost google-glog extra/jdk8-openjdk clang llvm thrift go cuda nvidia glew glfw libpng
    wget https://flexpp-bisonpp.googlecode.com/files/bisonpp-1.21-45.tar.gz
    tar xvf bisonpp-1.21-45.tar.gz
    cd bison++-1.21
    ./configure && make && make install

### CUDA

CUDA is installed to `/opt/cuda` instead of the default `/usr/local/cuda`. You may have to add the following to `CMakeLists.txt` to support this:

    include_directories("/opt/cuda/include")

# Environment variables

If using `mapd-deps-linux.sh` or Ubuntu 15.10, you will need to add following environment variables to your `~/.bashrc` or a file such as `/etc/profile.d/mapd-deps.sh` (or use [Environment Modules](http://modules.sourceforge.net/)):

    MAPD_PATH=/usr/local/mapd-deps
    PATH=$MAPD_PATH/bin:$PATH
    LD_LIBRARY_PATH=$MAPD_PATH/lib:$LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$MAPD_PATH/lib64:$LD_LIBRARY_PATH
    export PATH LD_LIBRARY_PATH

CUDA requires the following environment variables, assuming CUDA was installed to `/usr/local/cuda`:

    CUDA_PATH=/usr/local/cuda
    PATH=$CUDA_PATH/bin:$PATH
    LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
    export PATH LD_LIBRARY_PATH

CUDA on OS X is usually installed under `/Developer/NVIDIA/CUDA-7.5`:

    CUDA_PATH=/Developer/NVIDIA/CUDA-7.5
    PATH=$CUDA_PATH/bin:$PATH
    DYLD_LIBRARY_PATH=$CUDA_PATH/lib64:$DYLD_LIBRARY_PATH
    export PATH DYLD_LIBRARY_PATH
