#
# Dedalus Installation Script
#
# This script is designed to create a fully isolated Python installation
# with the dependencies you need to run dedalus.
#
# There are a few options, but you only need to set *one* of them.  And
# that's the next one, DEST_DIR.  But, if you want to use an existing HDF5
# installation you can set HDF5_DIR, or if you want to use some other
# subversion checkout of Dedalus, you can set DEDALUS_DIR, too.  (It'll already
# check the current directory and one up.
#

if [[ $1 == "-y" ]]; then
    INTERACTIVE=0
else
    INTERACTIVE=1
fi

if [[ $INTERACTIVE == 0 ]]; then
    CLEANUP=1
else
    CLEANUP=0
fi

DEST_SUFFIX="dedalus"
DEST_DIR="`pwd`/${DEST_SUFFIX/ /}"   # Installation location
BRANCH="tip" # This is the branch to which we will forcibly update.

MAKE_PROCS="-j4" # change this to set the number of cores you use to build packages

# If you need to supply arguments to the NumPy or SciPy build, supply them here
# This one turns on gfortran manually:
#NUMPY_ARGS="--fcompiler=gnu95"
# If you absolutely can't get the fortran to work, try this:
#NUMPY_ARGS="--fcompiler=fake"


# Packages

INST_OPENMPI=0 # by default, don't build OpenMPI. If you're on linux, use your package manager.
INST_HDF5=0 # by default, don't build HDF5.
INST_ATLAS=0 # by default, we will not build our own ATLAS. If you're on OSX, you'll want to use accelerate anyway.
INST_SCIPY=1
INST_IPYTHON=0 # by default, don't build ipython
INST_FTYPE=0 # by default, don't install freetype
INST_PNG=0 # by default, don't install libpng
INST_PKGCFG=0 # by default, don't install pkg-config
INST_OPENSSL=0 #by default, don't install openssl
INST_ZLIB=0 # by default, don't install zlib

if [ ${REINST_DEDALUS} ] && [ ${REINST_DEDALUS} -eq 1 ] && [ -n ${DEDALUS_DEST} ]
then
    DEST_DIR=${DEDALUS_DEST}
fi
# Make sure we are NOT being run as root
if [[ $EUID -eq 0 ]]
then
   echo "******************************************************"
   echo "*                                                    *"
   echo "*                                                    *"
   echo "*  IT IS A BAD IDEA TO RUN THIS SCRIPT AS ROOT!!!!   *"
   echo "*                                                    *"
   echo "*                                                    *"
   echo "******************************************************"
   echo
   echo "If you really want to do this, you must manually edit"
   echo "the script to re-enable root-level installation.  Sorry!"
   exit 1
fi
if [[ ${DEST_DIR%/} == /usr/local ]]
then
   echo "******************************************************"
   echo "*                                                    *"
   echo "*                                                    *"
   echo "*  THIS SCRIPT WILL NOT INSTALL TO /usr/local !!!!   *"
   echo "*                                                    *"
   echo "*                                                    *"
   echo "******************************************************"
   exit 1
fi

if type -P wget &>/dev/null
then
    echo "Using wget"
    export GETFILE="wget -nv"
else
    echo "Using curl"
    export GETFILE="curl -sSO"
fi

if type -P sha512sum &> /dev/null
then
    echo "Using sha512sum"
    export SHASUM="sha512sum"
elif type -P shasum &> /dev/null
then
    echo "Using shasum -a 512"
    export SHASUM="shasum -a 512"
else
    echo
    echo "Unable to locate any shasum-like utility."
    echo "ALL FILE INTEGRITY IS NOT VERIFIABLE."
    echo "THIS IS PROBABLY A BIG DEAL."
    echo
    echo "(I'll hang out for a minute for you to consider this.)"
    sleep 60
fi

function help_needed
{
    [ -e $1 ] && return
    echo
    echo "WE NEED YOUR HELP!"
    echo
    echo "We do not have testing facilities on $1 yet. While we're confident "
    echo "that Dedalus will work on $1, to get it working, "
    echo "we need to know the list of packages from $1 Dedalus requires."
    echo
    echo "If you are familiar with $1, please look over the package list for Ubuntu"
    echo "in this install script and help us translate it into $1 packages."
    echo
    echo "If you'd like to help, please don't hesitate to email the dev list,"
    echo "dedalus-dev@googlegroups.com"
    echo
    echo "   --the Dedalus team"
    echo
    echo
}

function get_dedalusproject
{
    [ -e $1 ] && return
    echo "Downloading $1 from dedalus-project.org"
    ${GETFILE} "http://data.dedalus-project.org/dependencies/$1" || do_exit
    ( ${SHASUM} -c $1.sha512 2>&1 ) 1>> ${LOG_FILE} || do_exit
}

function do_setup_py
{
    [ -e $1/done ] && return
    LIB=$1
    shift
    if [ -z "$@" ]
    then
        echo "Installing $LIB"
    else
        echo "Installing $LIB (arguments: '$@')"
    fi
    [ ! -e $LIB/extracted ] && tar xfz $LIB.tar.gz
    touch $LIB/extracted
    cd $LIB
    ( ${DEST_DIR}/bin/python3 setup.py build ${BUILD_ARGS} $* 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ${DEST_DIR}/bin/python3 setup.py install    2>&1 ) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
}

function do_exit
{
    echo "********************************************"
    echo "        FAILURE REPORT:"
    echo "********************************************"
    echo
    tail -n 10 ${LOG_FILE}
    echo
    echo "********************************************"
    echo "********************************************"
    echo "Failure.  Check ${LOG_FILE}.  The last 10 lines are above."
    exit 1
}

function host_specific
{
    IS_OSX=0              # not OSX by default
    MYHOST=`hostname -s`  # just give the short one, not FQDN
    MYOS=`uname -s`       # A guess at the OS
    if [ "${MYOS##Darwin}" != "${MYOS}" ]
    then
        echo "Looks like you're running on Mac OSX."
        echo
        echo "NOTE: you must have the Xcode command line tools installed."
	echo "You also need install mercurial (https://www.mercurial-scm.org/downloads), and gfortran (https://gcc.gnu.org/wiki/GFortranBinaries#MacOS)"
        echo
	echo "The instructions for obtaining the Xcode tools varies according"
	echo "to your exact OS version.  On older versions of OS X, you"
	echo "must register for an account on the apple developer tools"
	echo "website: https://developer.apple.com/downloads to obtain the"
	echo "download link."
	echo
	echo "We have gathered some additional instructions for each"
	echo "version of OS X below. If you have trouble installing Dedalus"
	echo "after following these instructions, don't hesitate to contact"
	echo "the Dedalus user's e-mail list."
	echo
	echo "You can see which version of OSX you are running by clicking"
	echo "'About This Mac' in the apple menu on the left hand side of"
	echo "menu bar.  We're assuming that you've installed all operating"
	echo "system updates; if you have an older version, we suggest"
	echo "running software update and installing all available updates."
        echo "OS X 10.7.5: download Xcode 4.2 from the mac app store"
	echo "(search for Xcode)."
        echo "Alternatively, download the Xcode command line tools from"
        echo "the Apple developer tools website."
        echo
	echo "OS X 10.8.4 and 10.9: download Xcode 5.02 from the mac app store."
	echo "(search for Xcode)."
    echo
	echo "Additionally, you will have to manually install the Xcode"
	echo "command line tools."
    echo
    echo "For OS X 10.8, see:"
   	echo "http://stackoverflow.com/questions/9353444"
	echo
    echo "For OS X 10.9, the command line tools can be installed"
    echo "with the following command:"
    echo "    xcode-select --install"
    echo
    echo "Once you have installed Xcode and the command line tools, install gfortran from https://gcc.gnu.org/wiki/GFortranBinaries#MacOS."

    OSX_VERSION=`sw_vers -productVersion`
    if [ "${OSX_VERSION##10.8}" != "${OSX_VERSION}" ]
        then
            MPL_SUPP_CFLAGS="${MPL_SUPP_CFLAGS} -mmacosx-version-min=10.7"
            MPL_SUPP_CXXFLAGS="${MPL_SUPP_CXXFLAGS} -mmacosx-version-min=10.7"
        fi
    if [ "${OSX_VERSION##10.9}" != "${OSX_VERSION}" ]
    then
	echo
        echo "Setting SDKROOT to address Xcode 6 issue in OSX 10.9."
        SDKROOT="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk"
	echo
    fi
    if [ "${OSX_VERSION##10.11}" != "${OSX_VERSION}" ]
    then
        INST_OPENSSL=1
	INST_ZLIB=1
    fi
    INST_OPENMPI=1
    INST_ATLAS=0
    INST_HDF5=1
    INST_FTYPE=1
    INST_PNG=1
    INST_PKGCFG=1
    IS_OSX=1
    elif [ -f /etc/redhat-release ]
    then
            echo "Looks like you're on a RedHat-compatible machine."
            echo
            echo "You need to have these packages installed:"
            echo
            echo "  * atlas"
            echo "  * atlas-devel"
            echo "  * mercurial"
            echo "  * openmpi"
            echo "  * openssl-devel"
            echo "  * ncurses"
            echo "  * ncurses-devel"
            echo "  * zip"
            echo "  * uuid"
            echo "  * uuid-devel"
            echo "  * freetype"
            echo "  * freetype-devel"
            echo "  * tk"
            echo "  * tk-devel"
            echo "  * hdf5"
            echo "  * hdf5-devel"
            #echo "  * zeromq"
            #echo "  * zeromq-devel"
	    echo "  * libpng-devel"
            echo "  * sqlite"
            echo "  * sqlite-devel"
            echo "  * gcc-gfortran"
            echo "  * gcc-c++"
            echo
            echo "You can accomplish this by executing:"
            echo
            echo "$ sudo yum install atlas atlas-devel openmpi openmpi-devel openssl openssl-devel ncurses ncurses-devel zip uuid uuid-devel freetype freetype-devel tk tk-devel hdf5 hdf5-devel libpng-devel mercurial sqlite sqlite-devel gcc-gfortran gcc-c++"
            echo
	    echo "Some of these packages may require access to EPEL (extra packages for enterprise linux), and you'll need optional packages enabled."
	    echo
	    echo " You will also need to add either /usr/lib/openmpi/bin or /usr/lib64/openmpi/bin to your PATH to access mpicc compilers."
	    echo
	    echo " $ setenv PATH /usr/lib64/openmpi/bin:$PATH"
	    echo " or "
	    echo " $ setenv PATH /usr/lib/openmpi/bin:$PATH"
            echo
            BLAS="/usr/lib/"
            LAPACK="/usr/lib/"
            export MPI_INCLUDE_PATH="/usr/include/openmpi-x86_64/"
            export MPI_LIBRARY_PATH="/usr/lib64/openmpi/lib/"
        echo
    elif [ -f /etc/lsb-release ] && [ `grep --count buntu /etc/lsb-release` -gt 0 ]
    then
        UBUNTU_VERSION=`lsb_release -r | cut -f 2 | sed -s 's/\([0-9]\+\)\.\([0-9]\+\)/\1/'`
        echo "Looks like you're on an Ubuntu-compatible machine."
        echo
        echo "You need to have these packages installed:"
        echo
        echo "  * libatlas-base-dev"
        echo "  * mercurial"
        echo "  * libatlas3-base"
        echo "  * libopenmpi-dev"
        echo "  * openmpi-bin"
        echo "  * libssl-dev"
        echo "  * build-essential"
        echo "  * libncurses5"
        echo "  * libncurses5-dev"
        echo "  * zip"
        echo "  * uuid-dev"
        echo "  * libfreetype6-dev"
        echo "  * tk-dev"
        echo "  * libhdf5-dev"
        if [ $UBUNTU_VERSION -lt 17 ]
        then
           echo "  * libzmq-dev"
        fi
        echo "  * libsqlite3-dev"
        echo "  * gfortran"
        echo
        echo "You can accomplish this by executing:"
        echo
        if [ $UBUNTU_VERSION -lt 17 ]
        then
            echo "$ sudo apt-get install libatlas-base-dev libatlas3-base libopenmpi-dev openmpi-bin libssl-dev build-essential libncurses5 libncurses5-dev zip uuid-dev libfreetype6-dev tk-dev libhdf5-dev mercurial libzmq-dev libsqlite3-dev gfortran"
        else
            echo "$ sudo apt-get install libatlas-base-dev libatlas3-base libopenmpi-dev openmpi-bin libssl-dev build-essential libncurses5 libncurses5-dev zip uuid-dev libfreetype6-dev tk-dev libhdf5-dev mercurial libsqlite3-dev gfortran"
        fi
        echo
        echo
        if [ $UBUNTU_VERSION -lt 16 ] 
        then
           echo "Your version of Ubuntu needs a newer version OpenMPI. We'll build our own."
           INST_OPENMPI=1
        fi
        if [ $UBUNTU_VERSION -ge 17 ]
        then
            echo "You are using Ubuntu 17 or higher, where the MPI headers have moved. Correcting header paths."
            export MPI_INCLUDE_PATH="/usr/lib/x86_64-linux-gnu/openmpi/include"
            export MPI_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/openmpi/lib"
        fi
        BLAS="/usr/lib/"
        LAPACK="/usr/lib/"
    elif [ -f /etc/debian_version ]
    then
	DEBIAN_VERSION=`cat /etc/debian_version`
        echo "Looks like you're on a Debian-compatible machine."
        echo
        echo "You need to have these packages installed:"
        echo
        echo "  * libatlas-base-dev"
        echo "  * mercurial"
        echo "  * libatlas3-base"
        echo "  * libopenmpi-dev"
        echo "  * openmpi-bin"
        echo "  * libssl-dev"
        echo "  * build-essential"
        echo "  * libncurses5"
        echo "  * libncurses5-dev"
        echo "  * zip"
        echo "  * uuid-dev"
        echo "  * libfreetype6-dev"
        echo "  * tk-dev"
        echo "  * libhdf5-dev"
        echo "  * libsqlite3-dev"
        echo "  * gfortran"
        echo
        echo "You can accomplish this by executing:"
        echo
        echo "$ sudo apt-get install libatlas-base-dev libatlas3-base libopenmpi-dev openmpi-bin libssl-dev build-essential libncurses5 libncurses5-dev zip uuid-dev libfreetype6-dev tk-dev libhdf5-dev mercurial libsqlite3-dev gfortran"
        echo
        echo
	echo "You're running Debian $DEBIAN_VERSION"
	if [ $(echo $DEBIAN_VERSION'<'9 | bc -l) -eq 1 ];
	then
            echo "Currently, Debian versions lower than 9 (Stretch) need a newer version OpenMPI. We'll build our own."
            INST_OPENMPI=1
	else
	    echo "You're running Debian >= 9; we can use the package provided OpenMPI."
	    export MPI_PATH=/usr/lib/x86_64-linux-gnu/openmpi
	fi
    fi

    # package installs
    if [ $INST_SCIPY -eq 1 ]
    then
	echo
	echo "Looks like you've requested that the install script build SciPy."
	echo
	echo "If the SciPy build fails, please uncomment one of the the lines"
	echo "at the top of the install script that sets NUMPY_ARGS, delete"
	echo "any broken installation tree, and re-run the install script"
	echo "verbatim."
	echo
	echo "If that doesn't work, don't hesitate to ask for help on the Dedalus"
	echo "user's mailing list."
	echo
    fi
    if [ ! -z "${CFLAGS}" ]
    then
        echo "******************************************"
        echo "******************************************"
        echo "**                                      **"
        echo "**    Your CFLAGS is not empty.         **"
        echo "**    This can break h5py compilation.  **"
        echo "**                                      **"
        echo "******************************************"
        echo "******************************************"
    fi
}
function do_setup_py
{
    [ -e $1/done ] && return
    LIB=$1
    shift
    if [ -z "$@" ]
    then
        echo "Installing $LIB"
    else
        echo "Installing $LIB (arguments: '$@')"
    fi
    [ ! -e $LIB/extracted ] && tar xfz $LIB.tar.gz
    touch $LIB/extracted
    cd $LIB
    if [ ! -z `echo $LIB | grep h5py` ]
    then
	( ${DEST_DIR}/bin/python3 setup.py build --hdf5=${HDF5_DIR} $* 2>&1 ) 1>> ${LOG_FILE} || do_exit
    else
        ( ${DEST_DIR}/bin/python3 setup.py build   $* 2>&1 ) 1>> ${LOG_FILE} || do_exit
    fi
    ( ${DEST_DIR}/bin/python3 setup.py install    2>&1 ) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
}

ORIG_PWD=`pwd`
if [ $INTERACTIVE == 1 ]; then
   echo "+++++++++"
   echo "Greetings, human. Welcome to the Dedalus Install Script"
   echo
fi
host_specific
if [ $INTERACTIVE == 1 ]; then
   echo
   echo
   read -p "[hit enter] "
   echo
   echo
fi

LOG_FILE="${DEST_DIR}/dedalus_install.log"

mkdir -p ${DEST_DIR}/src
cd ${DEST_DIR}/src

## Packages to install from source
PYTHON='Python-3.6.3'
FFTW='fftw-3.3.4'
NUMPY='numpy-1.11.0'
SCIPY='scipy-0.14.0'
OPENMPI='openmpi-1.10.1'
HDF5='hdf5-1.8.13'
FTYPE='freetype-2.5.3'
MATPLOTLIB='matplotlib-1.4.3'
PNG='libpng-1.6.17'
PKGCFG='pkg-config-0.28'
OPENSSL='openssl-1.0.1p'
ZLIB='zlib-1.2.8'

# dump sha512 to files
printf -v PYFILE "%s.tgz.sha512" $PYTHON
printf -v PYSHA "73f1477f3d3f5bd978c4ea1d1b679467b45e9fd2f443287b88c5c107a9ced580c56e0e8f33acea84e06b11a252e2a4e733120b721a9b6e1bb3d34493a3353bfb  %s" ${PYFILE%.sha512}

printf -v PYSHA "2cb33a20b77150ecbfc51f08195ec6d2f7a6f9c22a653544e657fb67be46dc8d6129c00a8bc51823476217dd09d290f1395acb63f1f07da50924553c8760a645  %s" ${PYFILE%.sha512}
echo "$PYSHA" > $PYFILE

printf -v FFTFILE "%s.tar.gz.sha512" $FFTW
printf -v FFTSHA "1ee2c7bec3657f6846e63c6dfa71410563830d2b951966bf0123bd8f4f2f5d6b50f13b76d9a7b0eae70e44856f829ca6ceb3d080bb01649d1572c9f3f68e8eb1  %s" ${FFTFILE%.sha512}
echo "$FFTSHA" > $FFTFILE

printf -v NPFILE "%s.tar.gz.sha512" $NUMPY
printf -v NPSHA "e3358b7b432bda76dedaad633319b2e242b187f91b3357574dbde2c4f5f8684a840e274e505b098ffd324f0dafa386939ef30b50fa89d6d901bf8e830fa47733  %s" ${NPFILE%.sha512}
echo "$NPSHA" > $NPFILE

printf -v SPFILE "%s.tar.gz.sha512" $SCIPY
printf -v SPSHA "ad1278740c1dc44c5e1b15335d61c4552b66c0439325ed6eeebc5872a1c0ba3fce1dd8509116b318d01e2d41da2ee49ec168da330a7fafd22511138b29f7235d  %s" ${SPFILE%.sha512}
echo "$SPSHA" > $SPFILE

printf -v MPIFILE "%s.tar.gz.sha512" $OPENMPI
printf -v MPISHA "9bac61e8cd2ddcca02d7053b7177d0d494eed43e1040d1532ab47eefb9bd14cdf7863a6460ccb859d98ab38458c03c240864084c41508a4743a986d2e95fb059  %s" ${MPIFILE%.sha512}
echo "$MPISHA" > $MPIFILE

printf -v HDF5FILE "%s.tar.gz.sha512" $HDF5
printf -v HDF5SHA "cd647ddf8cc6787cf57f3f84fd08b367158dc80f27669601a8c2fe573e14758c3c9d8787022a1c936d401c6676c1b4358b087825f46254342b0a35e06a2668be  %s" ${HDF5FILE%.sha512}
echo "$HDF5SHA" > $HDF5FILE

printf -v FTFILE "%s.tar.gz.sha512" $FTYPE
printf -v FTSHA "9ab7b77c5c09b1eb5baee7eb16da8a5f6fa7168cfa886bfed392b2fe80a985bcedecfbb8ed562c822ec9e48b061fb5fcdd9eea69eb44f970c2d1c55581f31d25  %s" ${FTFILE%.sha512}
echo "$FTSHA" > $FTFILE

printf -v MPLFILE "%s.tar.gz.sha512" $MATPLOTLIB
printf -v MPLSHA "51b0f58b2618b47b653e17e4f6b6a1215d3a3b0f1331ce3555cc7435e365d9c75693f289ce12fe3bf8f69fd57b663e545f0f1c2c94e81eaa661cac0689e125f5  %s" ${MPLFILE%.sha512}
echo "$MPLSHA" > $MPLFILE

printf -v PNGFILE "%s.tar.gz.sha512" $PNG
printf -v PNGSHA "445cf5cace57eb89f2f52be96e9f0e956717a4d4474bd6f5d0545a8b30b45ff45df94325c2504b044d014880cbb3e696475cd6fa7936993dee7ffee93756e384  %s" ${PNGFILE%.sha512}
echo "$PNGSHA" > $PNGFILE

printf -v PKGCFGFILE "%s.tar.gz.sha512" $PKGCFG
printf -v PKGCFGSHA "6eafa5ca77c5d44cd15f48457a5e96fcea2555b66d8e35ada5ab59864a0aa03d441e15f54ab9c6343693867b3b490f392c75b7d9312f024c9b7ec6a0194d8320  %s" ${PKGCFGFILE%.sha512}
echo "$PKGCFGSHA" > $PKGCFGFILE

printf -v SSLFILE "%s.tar.gz.sha512" $OPENSSL
printf -v SSLSHA "64e475c53a85b78de7c5aa71a22d4bb3a456142842373ebf8f22e9857cb0352b646e591b21af866933baecdbdb5ac4a22aeb64914440c53a0f30cd25914029e5  %s" ${SSLFILE%.sha512}
echo "$SSLSHA" > $SSLFILE

printf -v ZLIBFILE "%s.tar.gz.sha512" $ZLIB
printf -v ZLIBSHA "ece209d4c7ec0cb58ede791444dc754e0d10811cbbdebe3df61c0fd9f9f9867c1c3ccd5f1827f847c005e24eef34fb5bf87b5d3f894d75da04f1797538290e4a  %s" ${ZLIBFILE%.sha512}
echo "$ZLIBSHA" > $ZLIBFILE

# get the files
get_dedalusproject $PYTHON.tgz
get_dedalusproject $FFTW.tar.gz
get_dedalusproject $NUMPY.tar.gz
get_dedalusproject $SCIPY.tar.gz
[ $INST_OPENMPI -eq 1 ] && get_dedalusproject $OPENMPI.tar.gz
[ $INST_HDF5 -eq 1 ] && get_dedalusproject $HDF5.tar.gz
[ $INST_FTYPE -eq 1 ] && get_dedalusproject $FTYPE.tar.gz
[ $INST_PNG -eq 1 ]  && get_dedalusproject $PNG.tar.gz
[ $INST_PKGCFG -eq 1 ]  && get_dedalusproject $PKGCFG.tar.gz
[ $INST_OPENSSL -eq 1 ] && get_dedalusproject $OPENSSL.tar.gz
[ $INST_ZLIB -eq 1 ] && get_dedalusproject $ZLIB.tar.gz

# if we're installing freetype, we need to manually install matplotlib
[ $INST_FTYPE -eq 1 ] && get_dedalusproject $MATPLOTLIB.tar.gz

# first, OpenMPI, if we're doing that
if [ $INST_OPENMPI -eq 1 ]
then
    if [ ! -e $OPENMPI/done ]
    then
        [ ! -e $OPENMPI ] && tar xfz $OPENMPI.tar.gz
        echo "Installing OPENMPI"
        cd $OPENMPI
        ( ./configure CPPFLAGS=-I${DEST_DIR}/include CFLAGS=-I${DEST_DIR}/include --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    OPENMPI_DIR=${DEST_DIR}
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENMPI_DIR}/lib/"
    PATH="${OPENMPI_DIR}/bin:${PATH}"
    export MPI_PATH=${OPENMPI_DIR}
fi

# next, OpenSSL if we need it
if [ $INST_OPENSSL -eq 1 ]
then
    if [ ! -e $OPENSSL/done ]
    then
        [ ! -e $OPENSSL ] && tar xfz $OPENSSL.tar.gz
        echo "Installing OPENSSL"
        cd $OPENSSL
        ( ./Configure darwin64-x86_64-cc enable-ec_nistp_64_gcc_128 no-ssl2 no-ssl3 no-comp --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    OPENSSL_DIR=${DEST_DIR}
    export LDFLAGS="${LDFLAGS} -L${OPENSSL_DIR}/lib/ -L${OPENSSL_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENSSL_DIR}/lib/"
    PATH="${OPENSSL_DIR}/bin:${PATH}"
    export MPI_PATH=${OPENSSL_DIR}
fi

# next, zlib, if necessary
if [ $INST_ZLIB -eq 1 ]
then
    if [ ! -e $ZLIB/done ]
    then
        [ ! -e $ZLIB ] && tar xfz $ZLIB.tar.gz
        echo "Installing ZLIB"
        cd $ZLIB
        ( ./configure --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    ZLIB_DIR=${DEST_DIR}
    export LDFLAGS="${LDFLAGS} -L${ZLIB_DIR}/lib/ -L${ZLIB_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${ZLIB_DIR}/lib/"
    PATH="${ZLIB_DIR}/bin:${PATH}"
fi

# python3

if [ ! -e $PYTHON/done ]
then
    echo "Installing Python."
    [ ! -e $PYTHON ] && tar xfz $PYTHON.tgz
    cd $PYTHON
    if [ $INST_OPENSSL -eq 1 ]
    then
       export PY_CFLAGS="-I${DEST_DIR}/include"
    fi

    echo "PY_CFLAGS = $PY_CFLAGS"
    ( ./configure --prefix=${DEST_DIR}/ ${PYCONF_ARGS} CFLAGS=${PY_CFLAGS} 2>&1 ) 1>> ${LOG_FILE} || do_exit

    ( make ${MAKE_PROCS} 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
fi

export PYTHONPATH=${DEST_DIR}/lib/python3.5/site-packages/

# FFTW3
if [ ! -e $FFTW/done ]
then
    echo "Installing FFTW."
    [ ! -e $FFTW ] && tar xfz $FFTW.tar.gz
    cd $FFTW

    FFTWCONF_ARGS="CC=mpicc \
                   CXX=mpicxx \
                   F77=mpif90 \
                   MPICC=mpicc MPICXX=mpicxx \
                   --enable-shared \
                   --enable-mpi --enable-threads"

    ( ./configure --prefix=${DEST_DIR}/ ${FFTWCONF_ARGS} 2>&1 ) 1>> ${LOG_FILE} || do_exit

    ( make ${MAKE_PROCS} 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ln -sf ${DEST_DIR}/bin/python2.7 ${DEST_DIR}/bin/pyyt 2>&1 ) 1>> ${LOG_FILE}
    ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..

fi
export FFTW_PATH=${DEST_DIR}/

# HDF5, if we're doing that.
if [ $INST_HDF5 -eq 1 ]
then
    if [ ! -e $HDF5/done ]
    then
        [ ! -e $HDF5 ] && tar xfz $HDF5.tar.gz
        echo "Installing HDF5"
        cd $HDF5
        ( ./configure --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    export HDF5_DIR=${DEST_DIR}

fi

# freetype
if [ $INST_FTYPE -eq 1 ]
then
    if [ ! -e $FTYPE/done ]
    then
        [ ! -e $FTYPE ] && tar xfz $FTYPE.tar.gz
        echo "Installing FreeType2"
        cd $FTYPE
        ( ./configure CFLAGS=-I${DEST_DIR}/include --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
	( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    FTYPE_DIR=${DEST_DIR}
    export LDFLAGS="${LDFLAGS} -L${FTYPE_DIR}/lib/ -L${FTYPE_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${FTYPE_DIR}/lib/"
    export FTYPE_INST="$LDFLAGS"
fi


if [ $INST_PNG -eq 1 ]
then
    if [ ! -e $PNG/done ]
    then
        [ ! -e $PNG ] && tar xfz $PNG.tar.gz
        echo "Installing libpng"
        cd $PNG
        ( ./configure CFLAGS=-I${DEST_DIR}/include --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
	( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
    PNG_DIR=${DEST_DIR}
    export LDFLAGS="${LDFLAGS} -L${PNG_DIR}/lib/ -L${PNG_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PNG_DIR}/lib/"
    export PNG_INST="$LDFLAGS"
fi

if [ $INST_PKGCFG -eq 1 ]
then
    if [ ! -e $PKGCFG/done ]
    then
        [ ! -e $PKGCFG ] && tar xfz $PKGCFG.tar.gz
        echo "Installing pkg-config"
        cd $PKGCFG
        ( ./configure --with-internal-glib CFLAGS=-I${DEST_DIR}/include --prefix=${DEST_DIR}/ 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make 2>&1 ) 1>> ${LOG_FILE} || do_exit
	( make install CFLAGS=-std=gnu89 2>&1 ) 1>> ${LOG_FILE} || do_exit
        ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
        touch done
        cd ..
    fi
fi
# if !OSX ATLAS/OpenBLAS

# cython
echo "pip installing cython."
( ${DEST_DIR}/bin/pip3 install cython 2>&1 ) 1>> ${LOG_FILE} || do_exit

# numpy
# scipy
if [ ! -e $SCIPY/done ]
then

    # do some magic here...
    export BLAS=$BLAS
    export LAPACK=$LAPACK
    if [ $IS_OSX -eq 1 ]
    then
        export LDFLAGS="-bundle -undefined dynamic_lookup $LDFLAGS"
        export FFLAGS="$FFLAGS -fPIC"
        export FCFLAGS="$FCFLAGS -fPIC"
    fi
    do_setup_py $NUMPY ${NUMPY_ARGS}
    do_setup_py $SCIPY ${NUMPY_ARGS}
fi

# via pip:

# nose
echo "pip installing nose."
( ${DEST_DIR}/bin/pip3 install nose 2>&1 ) 1>> ${LOG_FILE} || do_exit

# mpi4py
echo "pip installing mpi4py."
( ${DEST_DIR}/bin/pip3 install mpi4py 2>&1 ) 1>> ${LOG_FILE} || do_exit

# h5py
echo "pip installing h5py."
( ${DEST_DIR}/bin/pip3 install h5py 2>&1 ) 1>> ${LOG_FILE} || do_exit

# matplotlib
PATH=$DEST_DIR/bin/:$PATH
if [ $INST_FTYPE -eq 1 ]
then
    echo "manually installing matplotlib."
# Now we set up the basedir for matplotlib:
    mkdir -p ${DEST_DIR}/src/$MATPLOTLIB
    echo "[directories]" >> ${DEST_DIR}/src/$MATPLOTLIB/setup.cfg
    echo "basedirlist = ${DEST_DIR}" >> ${DEST_DIR}/src/$MATPLOTLIB/setup.cfg
    if [ `uname` = "Darwin" ]
    then
	echo "[gui_support]" >> ${DEST_DIR}/src/$MATPLOTLIB/setup.cfg
	echo "macosx = False" >> ${DEST_DIR}/src/$MATPLOTLIB/setup.cfg
    fi

    [ ! -e $MATPLOTLIB/extracted ] && tar xfz $MATPLOTLIB.tar.gz
    touch $MATPLOTLIB/extracted
    cd $MATPLOTLIB
    patch -b setupext.py <<EOF
960c960
<             'freetype2', 'ft2build.h',
---
>             'freetype2', 'freetype2/ft2build.h',
EOF
    cd ..
    do_setup_py $MATPLOTLIB

else
    echo "pip installing matplotlib."
    ( ${DEST_DIR}/bin/pip3 install matplotlib 2>&1 ) 1>> ${LOG_FILE} || do_exit
fi

# ipython
if [ $INST_IPYTHON -eq 1 ]
then
    echo "pip installing ipython."
    ( ${DEST_DIR}/bin/pip3 install ipython 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ${DEST_DIR}/bin/pip3 install pyzmq 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ${DEST_DIR}/bin/pip3 install jinja2 2>&1 ) 1>> ${LOG_FILE} || do_exit

fi
# We assume that hg can be found in the path.
if type -P hg &>/dev/null
then
    export HG_EXEC=hg
else
    echo "Cannot find mercurial.  Please make sure it is installed."
    do_exit
fi

if [ -z "$DEDALUS_DIR" ]
then
    DEDALUS_DIR="$PWD/dedalus/"
    if [ ! -e dedalus ]
    then
        ( ${HG_EXEC} --debug clone https://bitbucket.org/dedalus-project/dedalus/ dedalus 2>&1 ) 1>> ${LOG_FILE}

        # Now we update to the branch we're interested in.
        ( ${HG_EXEC} -R ${DEDALUS_DIR} up -C ${BRANCH} 2>&1 ) 1>> ${LOG_FILE}
    fi
    echo Setting DEDALUS_DIR=${DEDALUS_DIR}
fi


## afterwards
# Add the environment scripts
( cp ${DEDALUS_DIR}/docs/activate ${DEST_DIR}/bin/activate 2>&1 ) 1>> ${LOG_FILE}
sed -i.bak -e "s,__DEDALUS_DIR__,${DEST_DIR}," ${DEST_DIR}/bin/activate
( cp ${DEDALUS_DIR}/docs/activate.csh ${DEST_DIR}/bin/activate.csh 2>&1 ) 1>> ${LOG_FILE}
sed -i.bak -e "s,__DEDALUS_DIR__,${DEST_DIR}," ${DEST_DIR}/bin/activate.csh

echo "Doing Dedalus update, wiping local changes and updating to branch ${BRANCH}"
MY_PWD=`pwd`
cd $DEDALUS_DIR
( ${HG_EXEC} pull 2>1 && ${HG_EXEC} up -C 2>1 ${BRANCH} 2>&1 ) 1>> ${LOG_FILE}

echo "Installing Dedalus"
( export PATH=$DEST_DIR/bin:$PATH ; ${DEST_DIR}/bin/python3 setup.py build_ext --inplace 2>&1 ) 1>> ${LOG_FILE} || do_exit
touch done
cd $MY_PWD

if !( ( ${DEST_DIR}/bin/python3 -c "import readline" 2>&1 )>> ${LOG_FILE})
then
    echo "Installing pure-python readline"
    ( ${DEST_DIR}/bin/pip3 install readline 2>&1 ) 1>> ${LOG_FILE}
fi

# clean up
if [ $CLEANUP == 1 ]; then
    rm ${DEST_DIR}/src/$PYTHON.tgz
    rm -rf ${DEST_DIR}/src/$PYTHON
    rm ${DEST_DIR}/src/$FFTW.tar.gz
    rm -rf ${DEST_DIR}/src/$FFTW
    rm ${DEST_DIR}/src/$NUMPY.tar.gz
    rm -rf ${DEST_DIR}/src/$NUMPY
    rm ${DEST_DIR}/src/$SCIPY.tar.gz
    rm -rf ${DEST_DIR}/src/$SCIPY
    [ $INST_OPENMPI -eq 1 ] && rm ${DEST_DIR}/src/$OPENMPI.tar.gz && rm -rf ${DEST_DIR}/src/$OPENMPI
    [ $INST_HDF5 -eq 1 ]    && rm ${DEST_DIR}/src/$HDF5.tar.gz    && rm -rf ${DEST_DIR}/src/$HDF5
    [ $INST_FTYPE -eq 1 ]   && rm ${DEST_DIR}/src/$FTYPE.tar.gz   && rm -rf ${DEST_DIR}/src/$FTYPE
    [ $INST_PNG -eq 1 ]     && rm ${DEST_DIR}/src/$PNG.tar.gz     && rm -rf ${DEST_DIR}/src/$PNG
    [ $INST_PKGCFG -eq 1 ]  && rm ${DEST_DIR}/src/$PKGCFG.tar.gz  && rm -rf ${DEST_DIR}/src/$PKGCFG
    [ $INST_OPENSSL -eq 1 ] && rm ${DEST_DIR}/src/$OPENSSL.tar.gz && rm -rf ${DEST_DIR}/src/$OPENSSL
    [ $INST_ZLIB -eq 1 ]    && rm ${DEST_DIR}/src/$ZLIB.tar.gz    && rm -rf ${DEST_DIR}/src/$ZLIB
fi

function print_afterword
{
    echo
    echo
    echo "========================================================================"
    echo
    echo "dedalus is now installed in $DEST_DIR ."
    echo
    echo "To run from this new installation, use the activate script for this "
    echo "environment."
    echo
    echo "    $ source $DEST_DIR/bin/activate"
    echo
    echo "This modifies the environment variables DEDALUS_DEST, PATH, PYTHONPATH, and"
    echo "LD_LIBRARY_PATH to activate dedalus.  If you use csh, just"
    echo "append .csh to the above."
    echo
    echo "The source for dedalus is located at:"
    echo "    $DEDALUS_DIR"
    echo
    echo "For support, see the website and join the mailing list:"
    echo
    echo "    http://dedalus-project.org/"
    echo "    http://dedalus-project.readthedocs.org/       (Docs)"
    echo
    echo "    https://groups.google.com/forum/#!forum/dedalus-users"
    echo
    echo "========================================================================"
    echo
    echo "Good luck, and email the user list if you run into any problems."
}

print_afterword
print_afterword >> ${LOG_FILE}

echo "dedalus dependencies were last updated on" > ${DEST_DIR}/.dedalus_update
date >> ${DEST_DIR}/.dedalus_update
