#===============================================================================
#
#  TypeProject: exe
#  Description:
#
#===============================================================================

ProjectName := cell_heat_test

#==============================CONFIGURATION: debug/release/miscellaneous...===#

ConfigurationName := release

#================================================================DIRECTORIES===#

ProjectDir := /home/primat/projects/tests/cell_heat_test

#==================================================INCDIRS (Format: IncDir*)===#

IncDirProjects = $(SupDirHome)
IncDirDealII   = $(SupDirDealII)/include
IncDirBoost    = $(SupDirDealII)/contrib/boost-1.49.0/include
IncDirTBB      = $(SupDirDealII)/contrib/tbb/tbb30_104oss/include 

#==================================================LIBDIRS (Format: LibDir*)===#

LibDirDealII = $(SupDirDealII)/lib

#===================================================USER DIRS (Format UDir*)===#

Udir = 

#===========================================SUPPORTING DIRS (Format SupDir*)===#

SupDirHome   = /home/primat
SupDirDealII = /home/primat/deal.II

#======================================================================FILES===#

Executable := $(ProjectName).exe

Sources    := $(ProjectName).cpp

DebugLibraryes    := deal_II.g tbb_debug CGAL_Core CGAL gmpxx mpfr gmp boost_thread-mt

ReleaseLibraryes  := deal_II tbb gomp pthread CGAL_Core CGAL gmpxx mpfr gmp boost_thread-mt z

#linux-vdso.so.1 =>  (0x00007fff80fff000)
#libgomp.so.1 => /usr/lib/x86_64-linux-gnu/libgomp.so.1 (0x00007fd324f64000)
#libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fd324bc5000)
#librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fd3249bc000)
#libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fd32479f000)
#/lib64/ld-linux-x86-64.so.2 (0x00007fd32519e000)

#=================================================================BASE FLAGS===#

DebugFlagsC   := -O0 -gstabs+ -pipe -std=c++11

ReleaseFlagsC := -std=c++11 -O2 -pipe -fopenmp -frounding-math -time #-ftree-parallelize-loops=2 #

DebugFlagL    := -gstabs+

ReleaseFlagL  := 

AnalysisFlags := -Weffc++ \
	             -Woverloaded-virtual -Wctor-dtor-privacy \
                 -Wnon-virtual-dtor -Wold-style-cast -Wconversion \
				 -Wsign-conversion -Winit-self -Wunreachable-code 

#=================================================================USER FLAGS===#

UserFlags :=

#==============================================================================#

ObjDir     := $(ProjectDir)/obj
SourcesDir := $(ProjectDir)/sources

CFlags :=
LFlags :=
ifeq ($(ConfigurationName), debug)
	CFlags := $(DebugFlagsC)
	LFlags := $(DebugFlagsL)
	Libs   := $(addprefix -l, $(DebugLibraryes))
endif
ifeq ($(ConfigurationName), release)
	CFlags := $(ReleaseFlagsC)
	LFlags := $(ReleaseFlagsL)
	Libs   := $(addprefix -l, $(ReleaseLibraryes))
endif

buff1   := $(filter IncDir%, $(.VARIABLES))
buff2   := $(foreach Dir, $(buff1), $($(Dir)))
IncDirs := $(addprefix -I, $(buff2)) -I.

buff1   := $(filter LibDir%, $(.VARIABLES))
buff2   := $(foreach Dir, $(buff1), $($(Dir)))
LibDirs := $(addprefix -L, $(buff2)) -L.


Objects    := $(patsubst %.cpp, %.o, $(Sources))
Objects    := $(addprefix $(ProjectDir)/obj/, $(Objects))

Sources    := $(addprefix $(ProjectDir)/sources/, $(Sources))

Executable := $(addprefix $(ProjectDir)/$(ConfigurationName)/, $(Executable)) 

#main:
#	@echo $(ObjDir)/Test1.o
	
$(Executable): $(Objects)
	$(CXX) $(LFlags) $^ -o $@ $(LibDirs) $(Libs)

$(ObjDir)/%.o: $(SourcesDir)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) -c $(CFlags) $< -o $@ $(IncDirs) $(LibDirs) $(Libs)  

analysis: $(Sources)
	$(CXX) $(AnalysisFlags) $^ $(IncDirs) $(LibDirs) $(Libs)  

run:
	$(Executable)

clean:
	$(RM)  $(Objects) $(Executable)

chead:
	$(CXX) -c $(CFlags) $(SourcesDir)/head.h -o $(SourcesDir)/head.h.gch $(IncDirs)

.PHONI: run clean
