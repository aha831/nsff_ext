# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dl/yulinquan3/nsff_pl/ceres-bin

# Include any dependencies generated for this target.
include examples/CMakeFiles/more_garbow_hillstrom.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/more_garbow_hillstrom.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/more_garbow_hillstrom.dir/flags.make

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o: examples/CMakeFiles/more_garbow_hillstrom.dir/flags.make
examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o: /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/more_garbow_hillstrom.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dl/yulinquan3/nsff_pl/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o -c /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/more_garbow_hillstrom.cc

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.i"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/more_garbow_hillstrom.cc > CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.i

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.s"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/more_garbow_hillstrom.cc -o CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.s

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.requires:

.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.requires

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.provides: examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.requires
	$(MAKE) -f examples/CMakeFiles/more_garbow_hillstrom.dir/build.make examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.provides.build
.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.provides

examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.provides.build: examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o


# Object files for target more_garbow_hillstrom
more_garbow_hillstrom_OBJECTS = \
"CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o"

# External object files for target more_garbow_hillstrom
more_garbow_hillstrom_EXTERNAL_OBJECTS =

bin/more_garbow_hillstrom: examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o
bin/more_garbow_hillstrom: examples/CMakeFiles/more_garbow_hillstrom.dir/build.make
bin/more_garbow_hillstrom: lib/libceres.a
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libglog.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libmetis.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libamd.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/more_garbow_hillstrom: /usr/local/cuda-11.4/lib64/libcudart_static.a
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/librt.so
bin/more_garbow_hillstrom: /usr/local/cuda-11.4/lib64/libcublas.so
bin/more_garbow_hillstrom: /usr/local/cuda-11.4/lib64/libcusolver.so
bin/more_garbow_hillstrom: /usr/local/cuda-11.4/lib64/libcusparse.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/more_garbow_hillstrom: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/more_garbow_hillstrom: examples/CMakeFiles/more_garbow_hillstrom.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dl/yulinquan3/nsff_pl/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/more_garbow_hillstrom"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/more_garbow_hillstrom.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/more_garbow_hillstrom.dir/build: bin/more_garbow_hillstrom

.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/build

examples/CMakeFiles/more_garbow_hillstrom.dir/requires: examples/CMakeFiles/more_garbow_hillstrom.dir/more_garbow_hillstrom.cc.o.requires

.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/requires

examples/CMakeFiles/more_garbow_hillstrom.dir/clean:
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && $(CMAKE_COMMAND) -P CMakeFiles/more_garbow_hillstrom.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/clean

examples/CMakeFiles/more_garbow_hillstrom.dir/depend:
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0 /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples /home/dl/yulinquan3/nsff_pl/ceres-bin /home/dl/yulinquan3/nsff_pl/ceres-bin/examples /home/dl/yulinquan3/nsff_pl/ceres-bin/examples/CMakeFiles/more_garbow_hillstrom.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/more_garbow_hillstrom.dir/depend

