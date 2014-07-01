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

DEST_SUFFIX="dedalus-`uname -m`"
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

function get_dedalusproject
{
    [ -e $1 ] && return
    echo "Downloading $1 from dedalus-project.org"
    ${GETFILE} "http://dedalus-project.org/dependencies/$1" || do_exit
    ( ${SHASUM} -c $1.sha512 2>&1 ) 1>> ${LOG_FILE} || do_exit
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
    MYHOST=`hostname -s`  # just give the short one, not FQDN
    MYHOSTLONG=`hostname` # FQDN, for Ranger
    MYOS=`uname -s`       # A guess at the OS
    if [ "${MYOS##Darwin}" != "${MYOS}" ]
    then
        echo "Looks like you're running on Mac OSX."
        echo
        echo "NOTE: you must have the Xcode command line tools installed."
        echo
	echo "The instructions for obtaining these tools varies according"
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
	echo
        echo "OS X 10.5.8: search for and download Xcode 3.1.4 from the"
	echo "Apple developer tools website."
        echo
        echo "OS X 10.6.8: search for and download Xcode 3.2 from the Apple"
	echo "developer tools website.  You can either download the"
	echo "Xcode 3.2.2 Developer Tools package (744 MB) and then use"
	echo "Software Update to update to XCode 3.2.6 or"
	echo "alternatively, you can download the Xcode 3.2.6/iOS SDK"
	echo "bundle (4.1 GB)."
        echo
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
    OSX_VERSION=`sw_vers -productVersion`
    if [ "${OSX_VERSION##10.8}" != "${OSX_VERSION}" ]
        then
            MPL_SUPP_CFLAGS="${MPL_SUPP_CFLAGS} -mmacosx-version-min=10.7"
            MPL_SUPP_CXXFLAGS="${MPL_SUPP_CXXFLAGS} -mmacosx-version-min=10.7"
        fi
    INST_OPENMPI=1
    INST_ATLAS=0
    fi

    if [ -f /etc/redhat-release ]
    then
        echo "Looks like you're on an Redhat-compatible machine."
        echo
        echo "You need to have these packages installed:"
        echo
        echo "  * openssl-devel"
        echo "  * uuid-devel"
        echo "  * readline-devel"
        echo "  * ncurses-devel"
        echo "  * zip"
        echo "  * gcc-{,c++,gfortran}"
        echo "  * make"
        echo "  * patch"
        echo 
        echo "You can accomplish this by executing:"
        echo "$ sudo yum install gcc gcc-g++ gcc-gfortran make patch zip"
        echo "$ sudo yum install ncurses-devel uuid-devel openssl-devel readline-devel"
    fi
    if [ -f /etc/SuSE-release ] && [ `grep --count SUSE /etc/SuSE-release` -gt 0 ]
    then
        echo "Looks like you're on an OpenSUSE-compatible machine."
        echo
        echo "You need to have these packages installed:"
        echo
        echo "  * devel_C_C++"
        echo "  * libopenssl-devel"
        echo "  * libuuid-devel"
        echo "  * zip"
        echo "  * gcc-c++"
        echo
        echo "You can accomplish this by executing:"
        echo
        echo "$ sudo zypper install -t pattern devel_C_C++"
        echo "$ sudo zypper install gcc-c++ libopenssl-devel libuuid-devel zip"
        echo
        echo "I am also setting special configure arguments to Python to"
        echo "specify control lib/lib64 issues."
        PYCONF_ARGS="--libdir=${DEST_DIR}/lib"
    fi
    if [ -f /etc/lsb-release ] && [ `grep --count buntu /etc/lsb-release` -gt 0 ]
    then
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
        echo
        echo "You can accomplish this by executing:"
        echo
        echo "$ sudo apt-get install libatlas-base-dev libatlas3-base libopenmpi-dev openmpi-bin libssl-dev build-essential libncurses5 libncurses5-dev zip uuid-dev libfreetype6-dev tk-dev libhdf5-dev mercurial"
        echo
        echo
        BLAS="/usr/lib/"
        LAPACK="/usr/lib/"
    fi
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
echo "+++++++++"
echo "Greetings, human. Welcome to the Dedalus Install Script"
echo
host_specific
echo
echo
read -p "[hit enter] "
echo
echo

LOG_FILE="${DEST_DIR}/dedalus_install.log"

mkdir -p ${DEST_DIR}/src
cd ${DEST_DIR}/src

## Packages to install from source
PYTHON='Python-3.4.1'
FFTW='fftw-3.3.4'
NUMPY='numpy-1.8.1'
SCIPY='scipy-0.14.0'
OPENMPI='openmpi-1.6.5'

# dump sha512 to files
printf -v PYFILE "%s.tgz.sha512" $PYTHON
printf -v PYSHA "43adbef25e1b7a7a0be86231d4aa131d5aa3efd5608f572b3b0878520cd27054cc7993726f7ba4b5c878e0c4247fb443f367e0a786dec9fbfa7a25d67427f107 %s" ${PYFILE%.sha512}
echo $PYSHA > $PYFILE

printf -v FFTFILE "%s.tar.gz.sha512" $FFTW
printf -v FFTSHA "1ee2c7bec3657f6846e63c6dfa71410563830d2b951966bf0123bd8f4f2f5d6b50f13b76d9a7b0eae70e44856f829ca6ceb3d080bb01649d1572c9f3f68e8eb1 %s" ${FFTFILE%.sha512}
echo $FFTSHA > $FFTFILE

printf -v NPFILE "%s.tar.gz.sha512" $NUMPY
printf -v NPSHA "39ef9e13f8681a2c2ba3d74ab96fd28c5669e653308fd1549f262921814fa7c276ce6d9fb65ef135006584c608bdf3db198d43f66c9286fc7b3c79803dbc1f57 %s" ${NPFILE%.sha512}
echo $NPSHA > $NPFILE

printf -v SPFILE "%s.tar.gz.sha512" $SCIPY
printf -v SPSHA "ad1278740c1dc44c5e1b15335d61c4552b66c0439325ed6eeebc5872a1c0ba3fce1dd8509116b318d01e2d41da2ee49ec168da330a7fafd22511138b29f7235d %s" ${SPFILE%.sha512}
echo $SPSHA > $SPFILE

# get the files
get_dedalusproject $PYTHON.tgz
get_dedalusproject $FFTW.tar.gz
get_dedalusproject $NUMPY.tar.gz
get_dedalusproject $SCIPY.tar.gz
[ $INST_OPENMPI -eq 1 ] && get_dedalusproject $OPENMPI.tar.gz


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
    export LDFLAGS="${LDFLAGS} -L${OPENMPI_DIR}/lib/ -L${OPENMPI_DIR}/lib64/"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${OPENMPI_DIR}/lib/"
    PATH="${OPENMPI_DIR}/bin:${PATH}"
fi


# python3 

if [ ! -e $PYTHON/done ]
then
    echo "Installing Python."
    [ ! -e $PYTHON ] && tar xfz $PYTHON.tgz
    cd $PYTHON
    ( ./configure --prefix=${DEST_DIR}/ ${PYCONF_ARGS} 2>&1 ) 1>> ${LOG_FILE} || do_exit

    ( make ${MAKE_PROCS} 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..
fi

export PYTHONPATH=${DEST_DIR}/lib/python3.4/site-packages/

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
                   --enable-mpi --enable-openmp --enable-threads"

    ( ./configure --prefix=${DEST_DIR}/ ${FFTWCONF_ARGS} 2>&1 ) 1>> ${LOG_FILE} || do_exit

    ( make ${MAKE_PROCS} 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( make install 2>&1 ) 1>> ${LOG_FILE} || do_exit
    ( ln -sf ${DEST_DIR}/bin/python2.7 ${DEST_DIR}/bin/pyyt 2>&1 ) 1>> ${LOG_FILE}
    ( make clean 2>&1) 1>> ${LOG_FILE} || do_exit
    touch done
    cd ..

fi
export FFTW_PATH=${DEST_DIR}/

# if !OSX ATLAS/OpenBLAS

# numpy
# scipy
if [ ! -e $SCIPY/done ]
then

    # do some magic here...
    export BLAS=$BLAS
    export LAPACK=$LAPACK
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

# cython
echo "pip installing cython."
( ${DEST_DIR}/bin/pip3 install cython 2>&1 ) 1>> ${LOG_FILE} || do_exit

# matplotlib
echo "pip installing matplotlib."
( ${DEST_DIR}/bin/pip3 install -v https://downloads.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.3.1/matplotlib-1.3.1.tar.gz 2>&1 ) 1>> ${LOG_FILE} || do_exit

# sympy
echo "pip installing sympy."
( ${DEST_DIR}/bin/pip3 install sympy 2>&1 ) 1>> ${LOG_FILE} || do_exit

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
    if [ ! -e dedalus2 ]
    then
        DEDALUS_DIR="$PWD/dedalus2/"
        ( ${HG_EXEC} --debug clone https://bitbucket.org/jsoishi/dedalus2-jsoishi/ dedalus2 2>&1 ) 1>> ${LOG_FILE}
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

echo "Doing yt update, wiping local changes and updating to branch ${BRANCH}"
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
    ( ${DEST_DIR}/bin/pip install readline 2>&1 ) 1>> ${LOG_FILE}
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
    echo "LD_LIBRARY_PATH to match your new yt install.  If you use csh, just"
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
