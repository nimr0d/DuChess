# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_SOURCE_DIR = /home/nimrod/Code/chess_cv

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nimrod/Code/chess_cv

# Include any dependencies generated for this target.
include CMakeFiles/chess_cv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/chess_cv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/chess_cv.dir/flags.make

CMakeFiles/chess_cv.dir/src/main.cc.o: CMakeFiles/chess_cv.dir/flags.make
CMakeFiles/chess_cv.dir/src/main.cc.o: src/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nimrod/Code/chess_cv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/chess_cv.dir/src/main.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/chess_cv.dir/src/main.cc.o -c /home/nimrod/Code/chess_cv/src/main.cc

CMakeFiles/chess_cv.dir/src/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/chess_cv.dir/src/main.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nimrod/Code/chess_cv/src/main.cc > CMakeFiles/chess_cv.dir/src/main.cc.i

CMakeFiles/chess_cv.dir/src/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/chess_cv.dir/src/main.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nimrod/Code/chess_cv/src/main.cc -o CMakeFiles/chess_cv.dir/src/main.cc.s

CMakeFiles/chess_cv.dir/src/main.cc.o.requires:

.PHONY : CMakeFiles/chess_cv.dir/src/main.cc.o.requires

CMakeFiles/chess_cv.dir/src/main.cc.o.provides: CMakeFiles/chess_cv.dir/src/main.cc.o.requires
	$(MAKE) -f CMakeFiles/chess_cv.dir/build.make CMakeFiles/chess_cv.dir/src/main.cc.o.provides.build
.PHONY : CMakeFiles/chess_cv.dir/src/main.cc.o.provides

CMakeFiles/chess_cv.dir/src/main.cc.o.provides.build: CMakeFiles/chess_cv.dir/src/main.cc.o


# Object files for target chess_cv
chess_cv_OBJECTS = \
"CMakeFiles/chess_cv.dir/src/main.cc.o"

# External object files for target chess_cv
chess_cv_EXTERNAL_OBJECTS =

chess_cv: CMakeFiles/chess_cv.dir/src/main.cc.o
chess_cv: CMakeFiles/chess_cv.dir/build.make
chess_cv: /usr/local/lib/libopencv_shape.so.3.1.0
chess_cv: /usr/local/lib/libopencv_stitching.so.3.1.0
chess_cv: /usr/local/lib/libopencv_superres.so.3.1.0
chess_cv: /usr/local/lib/libopencv_videostab.so.3.1.0
chess_cv: /usr/local/lib/libopencv_objdetect.so.3.1.0
chess_cv: /usr/local/lib/libopencv_calib3d.so.3.1.0
chess_cv: /usr/local/lib/libopencv_features2d.so.3.1.0
chess_cv: /usr/local/lib/libopencv_flann.so.3.1.0
chess_cv: /usr/local/lib/libopencv_highgui.so.3.1.0
chess_cv: /usr/local/lib/libopencv_ml.so.3.1.0
chess_cv: /usr/local/lib/libopencv_photo.so.3.1.0
chess_cv: /usr/local/lib/libopencv_video.so.3.1.0
chess_cv: /usr/local/lib/libopencv_videoio.so.3.1.0
chess_cv: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
chess_cv: /usr/local/lib/libopencv_imgproc.so.3.1.0
chess_cv: /usr/local/lib/libopencv_core.so.3.1.0
chess_cv: CMakeFiles/chess_cv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nimrod/Code/chess_cv/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable chess_cv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/chess_cv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/chess_cv.dir/build: chess_cv

.PHONY : CMakeFiles/chess_cv.dir/build

CMakeFiles/chess_cv.dir/requires: CMakeFiles/chess_cv.dir/src/main.cc.o.requires

.PHONY : CMakeFiles/chess_cv.dir/requires

CMakeFiles/chess_cv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/chess_cv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/chess_cv.dir/clean

CMakeFiles/chess_cv.dir/depend:
	cd /home/nimrod/Code/chess_cv && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nimrod/Code/chess_cv /home/nimrod/Code/chess_cv /home/nimrod/Code/chess_cv /home/nimrod/Code/chess_cv /home/nimrod/Code/chess_cv/CMakeFiles/chess_cv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/chess_cv.dir/depend

