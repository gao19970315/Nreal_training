# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /opt/cmake-3.13.0/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.13.0/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/Desktop/Nreal/task3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/Desktop/Nreal/task3/build

# Include any dependencies generated for this target.
include CMakeFiles/1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/1.dir/flags.make

CMakeFiles/1.dir/src.cpp.o: CMakeFiles/1.dir/flags.make
CMakeFiles/1.dir/src.cpp.o: ../src.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/Desktop/Nreal/task3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/1.dir/src.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/1.dir/src.cpp.o -c /root/Desktop/Nreal/task3/src.cpp

CMakeFiles/1.dir/src.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/1.dir/src.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/Desktop/Nreal/task3/src.cpp > CMakeFiles/1.dir/src.cpp.i

CMakeFiles/1.dir/src.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/1.dir/src.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/Desktop/Nreal/task3/src.cpp -o CMakeFiles/1.dir/src.cpp.s

# Object files for target 1
1_OBJECTS = \
"CMakeFiles/1.dir/src.cpp.o"

# External object files for target 1
1_EXTERNAL_OBJECTS =

1 : CMakeFiles/1.dir/src.cpp.o
1 : CMakeFiles/1.dir/build.make
1 : CMakeFiles/1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/Desktop/Nreal/task3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable 1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/1.dir/build: 1

.PHONY : CMakeFiles/1.dir/build

CMakeFiles/1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/1.dir/clean

CMakeFiles/1.dir/depend:
	cd /root/Desktop/Nreal/task3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/Desktop/Nreal/task3 /root/Desktop/Nreal/task3 /root/Desktop/Nreal/task3/build /root/Desktop/Nreal/task3/build /root/Desktop/Nreal/task3/build/CMakeFiles/1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/1.dir/depend
