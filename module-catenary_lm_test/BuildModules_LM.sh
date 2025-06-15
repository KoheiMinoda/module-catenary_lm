# ! /bin/bash


cd mbdyn
rm -f *.o
cd ../

cd modules/module-catenary_lm
rm -rf .libs
rm -f *.la *.lo *.o
cd ../../

# build modules
LDFLAGS=-rdynamic
LIBS=/usr/lib/x86_64-linux-gnu/libltdl.a
CC="gcc-9" CXX="g++-9" F77="gfortran-9" FC="gfortran-9" CPPFLAGS=-I/usr/include/suitesparse ./configure --enable-runtime-loading 
CC="gcc-9" CXX="g++-9" F77="gfortran-9" FC="gfortran-9" CPPFLAGS=-I/usr/include/suitesparse ./configure --with-module="catenary_lm"
make
sudo make install

# delete .mod file for viewing
cd modules/module-catenary_lm
rm -f *.mod
cd ../

# install ufmpack
CPPFLAGS=-I/usr/include/suitesparse ./configure
make
sudo make install
