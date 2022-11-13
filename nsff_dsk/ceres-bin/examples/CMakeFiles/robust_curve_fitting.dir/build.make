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
include examples/CMakeFiles/robust_curve_fitting.dir/depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/robust_curve_fitting.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/robust_curve_fitting.dir/flags.make

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o: examples/CMakeFiles/robust_curve_fitting.dir/flags.make
examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o: /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/robust_curve_fitting.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dl/yulinquan3/nsff_pl/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o -c /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/robust_curve_fitting.cc

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.i"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/robust_curve_fitting.cc > CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.i

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.s"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples/robust_curve_fitting.cc -o CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.s

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.requires:

.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.requires

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.provides: examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.requires
	$(MAKE) -f examples/CMakeFiles/robust_curve_fitting.dir/build.make examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.provides.build
.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.provides

examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.provides.build: examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o


# Object files for target robust_curve_fitting
robust_curve_fitting_OBJECTS = \
"CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o"

# External object files for target robust_curve_fitting
robust_curve_fitting_EXTERNAL_OBJECTS =

bin/robust_curve_fitting: examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o
bin/robust_curve_fitting: examples/CMakeFiles/robust_curve_fitting.dir/build.make
bin/robust_curve_fitting: lib/libceres.a
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libglog.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libspqr.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libcholmod.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libmetis.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libamd.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libcamd.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libccolamd.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libcolamd.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libcxsparse.so
bin/robust_curve_fitting: /usr/local/cuda-11.4/lib64/libcudart_static.a
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/librt.so
bin/robust_curve_fitting: /usr/local/cuda-11.4/lib64/libcublas.so
bin/robust_curve_fitting: /usr/local/cuda-11.4/lib64/libcusolver.so
bin/robust_curve_fitting: /usr/local/cuda-11.4/lib64/libcusparse.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/liblapack.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libf77blas.so
bin/robust_curve_fitting: /usr/lib/x86_64-linux-gnu/libatlas.so
bin/robust_curve_fitting: examples/CMakeFiles/robust_curve_fitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dl/yulinquan3/nsff_pl/ceres-bin/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/robust_curve_fitting"
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/robust_curve_fitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/robust_curve_fitting.dir/build: bin/robust_curve_fitting

.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/build

examples/CMakeFiles/robust_curve_fitting.dir/requires: examples/CMakeFiles/robust_curve_fitting.dir/robust_curve_fitting.cc.o.requires

.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/requires

examples/CMakeFiles/robust_curve_fitting.dir/clean:
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin/examples && $(CMAKE_COMMAND) -P CMakeFiles/robust_curve_fitting.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/clean

examples/CMakeFiles/robust_curve_fitting.dir/depend:
	cd /home/dl/yulinquan3/nsff_pl/ceres-bin && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0 /home/dl/yulinquan3/nsff_pl/ceres-solver-2.1.0/examples /home/dl/yulinquan3/nsff_pl/ceres-bin /home/dl/yulinquan3/nsff_pl/ceres-bin/examples /home/dl/yulinquan3/nsff_pl/ceres-bin/examples/CMakeFiles/robust_curve_fitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/robust_curve_fitting.dir/depend
