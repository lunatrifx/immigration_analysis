cmake_minimum_required(VERSION 3.18 FATAL_ERROR) # Increase to 3.18 or higher for full CUDA language support

# Set policies to suppress dev warnings (good practice for current project)
# You can uncomment these to get rid of the warnings,
# but understanding them is better than just suppressing.
cmake_policy(SET CMP0146 NEW) # To suppress the FindCUDA removal warning
cmake_policy(SET CMP0104 NEW) # To suppress the empty CMAKE_CUDA_ARCHITECTURES warning

project(ImmigrationAnalysis LANGUAGES CXX CUDA) # <-- CRITICAL: Declare CUDA as a language here!

# Specify CUDA architectures
# You can list multiple if you need to support different GPU generations.
# Common ones: sm_35 (Kepler), sm_50 (Maxwell), sm_60 (Pascal), sm_70 (Volta), sm_75 (Turing), sm_80 (Ampere), sm_86 (Ampere), sm_87 (Ampere), sm_89 (Ada Lovelace), sm_90 (Hopper)
# You can check your GPU's compute capability if you're unsure.
# For simplicity, let CMake auto-detect or pick a recent common one.
# If you want auto-detection (requires CMake 3.18+):
# set(CMAKE_CUDA_ARCHITECTURES NATIVE)
# Or for specific architectures:
set(CMAKE_CUDA_ARCHITECTURES "89;80") # Example: for Ampere GPUs (RTX 30 series, A100)
                                    # Choose architectures relevant to your target GPUs.
                                    # If you only have one specific GPU, use its compute capability.
                                    # For most modern cards, "86" is a good start.

add_executable(ImmigrationAnalysis
    src/main.cpp
    src/cuda_analyzer.cu # Add your .cu files directly here
    src/data_loader.cpp
    src/visualization_manager.cpp

    # ... other source files
)



# Link your executable. When CUDA is declared as a language in project(),
# CMake automatically finds and configures the CUDA runtime library.
# You typically DO NOT need to explicitly link CUDA::cudart_static or similar
# unless you have very specific needs or are linking other CUDA libraries (e.g., cuFFT, cuBLAS).
# CMake's built-in CUDA language support handles the basic runtime linking.
target_link_libraries(ImmigrationAnalysis
    # ... any other non-CUDA libraries your project needs (e.g., OpenCV, Boost)
    # If you *still* need to explicitly link `cudart_static` for some reason,
    # it should now be found by CMake since you declared CUDA as a language.
    # CUDA::cudart_static # You might not need this line if simple CUDA files are added.
)