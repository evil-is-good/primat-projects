#===============================================================================
#
#  TypeProject: exe
#  Description:
#
#===============================================================================

ProjectName := complete_solution_of_heat_problem

#==============================CONFIGURATION: debug/release/miscellaneous...===#

ConfigurationName := debug

#================================================================DIRECTORIES===#

ProjectDir := /home/primat/projects/tests/complete_solution_of_heat_problem

#==================================================INCDIRS (Format: IncDir*)===#

IncDirProjects = $(SupDirHome)
IncDirDealII   = $(SupDirDealII)/include
IncDirBoost    = $(SupDirDealII)/contrib/boost-1.46.1/include
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

DebugLibraryes    := deal_II.g tbb_debug

ReleaseLibraryes  := deal_II tbb

#=================================================================BASE FLAGS===#

DebugFlagsC   := -O0 -gstabs+ -pipe -std=c++0x

ReleaseFlagsC := -O3 -pipe

DebugFlagL    := -gstabs+

ReleaseFlagL  := 

AnalysisFlags := #-Weffc++ \
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

.PHONI: run clean
