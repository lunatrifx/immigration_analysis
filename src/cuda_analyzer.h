#pragma once
#include "data_loader.h"
#include <vector>
#include <map>
#include <iostream>

// Struct for incident data that can be processed on GPU
struct DeviceIncident {
    // We only need the data relevent for GPU processign (lat, long, border, more later)
    double Latitude;
    double Longitude;
    int border_type; // 0 for US-Canada, 1 for US-Mexico
    // More fields can be added as needed
};

class ImmigrationIncident;
// Host-side CUDA Analyzer class
class CUDAAnalyzer {
    public:
    CUDAAnalyzer(); // Default Constructor
    ~CUDAAnalyzer(); // Destructor
    // Analyze incidents and returns count per state
    std::map<std::string, int> analyzeIncidentsByState(const std::vector<ImmigrationIncident>& incidents);
        std::map<std::string, int> StateIncidentCount;

    // Placeholder for future GPU-based geospatial heatmap analysis
    // static std::vector<GeoGridCell> generateHeatmap(const std::vector<ImmigrationIncident>& incidents, int gridSize
};

// Helper for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)
