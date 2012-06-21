include Makefile.$(HOSTNAME).in

#
# where to look for include files
#
INCLS  = -I. -I$(LIFEINCLUDEPATH) -I$(TRILINCLUDEPATH) -I$(PARMETISINCLUDEPATH) -I$(HDF5INCLUDEPATH) \
 -I$(BOOSTINCLUDEPATH) -I$(MPIINCLUDEPATH) -I$(BLASINCLUDEPATH) -I$(UMFPACKINCLUDEPATH)
#
# Preprocessor and compilation flags
#
CXXFLAGS=$(OPTFLAGS) $(WARNFLAGS) -ansi -m64
CPPFLAGS=$(INCLS) -DTRILINOS_AZTEC -DNDEBUG -DNDEBUG_OLD # when compiling in opt mode
# CPPFLAGS=$(INCLS) -DTRILINOS_AZTEC -DLIFEV_CHECK_ALL # when compiling in debug mode
#
# Additional libraries
LOADLIBES = $(LIFELIBS) $(TRILLIBS) $(PARMETISLIBS) $(HDF5LIBS) \
 $(MPILIBS) $(UMFPACKLIBS) $(BLASLIBS)
#
#Linker flags
#
LDFLAGS += $(CXXFLAGS) $(LIFELDFLAGS) $(TRILLDFLAGS) $(PARMETISLDFLAGS) \
 $(HDF5LDFLAGS) $(MPILDFLAGS) $(BLASLDFLAGS) $(UMFPACKLDFLAGS)
#
# get all files *.cpp and identify the main
#
SRCS=$(wildcard *.cpp)
EXE_SRC=$(filter compute%.cpp,$(SRCS))
OTHER_SRCS=$(filter-out $(EXE_SRC),$(SRCS))
#
# get the corresponding object file
#
MAIN_OBJ=$(EXE_SRC:.cpp=.o)
OTHER_OBJS = $(OTHER_SRCS:.cpp=.o)
OBJS= $(MAIN_OBJ) $(OTHER_OBJS)
#
# name of the file containing include dependencies
#
DEPEND=make.dep
#
# 
# Call the executable with the same name as the main file
#
EXEC=$(EXE_SRC:.cpp=)


#========================== TARGET DEFINITIONS
.phony= all  clean cleantemp veryclean

all: $(DEPEND) $(EXEC)

clean:
	-\rm -f $(EXEC) $(OBJS) $(DEPEND)

cleantemp:
	-\rm -f *~ *.m *.scl *.vct *.case *.geo *.h5 *.xmf

veryclean: clean cleantemp

#
# We assume that the main program depends on all
# the object files
#

$(EXEC): $(OBJS)

$(OBJS): $(SRCS) 

#
#generate dependencies (silently)
#
$(DEPEND): $(SRCS)
	@g++ $(CPPFLAGS) -MM $(SRCS) -MF $(DEPEND)

-include $(DEPEND)
