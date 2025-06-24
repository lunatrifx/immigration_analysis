#include <iostream>
#include "cuda_analyzer.h"
#include <cuda_runtime.h>
#include <unordered_map> // for mapping state names to IDs on the host
#include <device_launch_parameters.h> // for CUDA kernel launch parameters
#include <vector>
#include <map>

// Define a max number of states
// For a truly robust implementation, this should be dynamic. Need more complex strategies (such as hash maps on devices or multiple kernel launches)
const int MAX_STATES = 60; // US States, DC, territories, Canada/Mexico

// Simple device side struct for incident data
// in plain C++, for CUDA
struct DeviceIncident_GPU {
    double Latitude;
    double Longitude;
    int border_type; // 0: Canada, 1: Mexico
    int state_id; // Mapped state ID aka integer ID for the state

    DeviceIncident_GPU(double lat = 0.0, double lon = 0.0, int border = 0, int state = -1)
        : Latitude(lat), Longitude(lon), border_type(border), state_id(state) {}
};

// Constructor Definition
CUDAAnalyzer::CUDAAnalyzer() {
    // Constructor can be used to initialize any CUDA resources if needed
}

// Destructor Definition
CUDAAnalyzer::~CUDAAnalyzer() {
    // Destructor can be used to clean up any CUDA resources if needed
    // Currently, no dynamic resources are allocated in this class
}

// analyzeIncidentsByState function definition
std::map<std::string, int> CUDAAnalyzer::analyzeIncidentsByState(const std::vector<ImmigrationIncident>& incidents) {
   std::map<std::string, int> hostStateCounts;
   std::map<std::string, int> borderCounts;

   for (const auto& incident : incidents) {
    // Logic to normalize and count by Border
    // *** Special debug *** 
    std::cout << "DEBUG: CUDAAnalyzer processing border: '" << incident.Border << "'" << std::endl;
    // *** End Special debug ***
    if (incident.Border == "US-Canada Border") {
        borderCounts["US-Canada Border"]++; 
    } 
    else if (incident.Border == "US-Mexico Border") {
        borderCounts["US-Mexico Border"]++;
    } 
    else if (incident.Border == "Other Borders") {
        borderCounts["Other Borders"]++;
        std::cout << "Unknown border type: '" << incident.Border << "'" << std::endl;
    } 
    else {
        borderCounts["Unknown Border"]++;
    }

    // Placeholder for State Derivation 

    // *** We'll also do a debug here *** 
    std::cout << "DEBUG: CUDAAnalyzer processing state: '" << incident.State << "'" << std::endl;
    // *** End Special debug ***
    if (!incident.State.empty()) {
        hostStateCounts[incident.State]++;
    } else {
        hostStateCounts["Unknown State"]++;
    }
   }

    // Print Border counts
    std::cout << "\nLegal Immigration Numbers by Border:" << std::endl;
    if (borderCounts.empty()) {
        std::cout << "No border data generated." << std::endl;
    } 
    else 
    {
        for (const auto& pair : borderCounts) {
            std::cout << "Border: " << pair.first << ", Count: " << pair.second << std::endl;
        }
    }

    // Print State Counts
    std::cout << "\nLegal Immigration Numbers by Individual State:" << std::endl;
    if (hostStateCounts.empty()) {
        std::cout << "No state data found." << std::endl;
   } 
    else 
    {
        for (const auto& pair : hostStateCounts) {
            std::cout << "State: " << pair.first << ", Count: " << pair.second << std::endl;
        }
    }

    return hostStateCounts; // Return the state counts
}
 

// Creating the global array on the device to store state counts
__device__ int g_stateCounts[MAX_STATES];

// Initializing a CUDA kernel to count the incidents per state
__global__ void countIncidentsByStateKernel(DeviceIncident_GPU* d_incidents, int numIncidents)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numIncidents) {
        // This is assumning state_id is already mapped correctly on the host
        if (d_incidents[idx].state_id >= 0 && d_incidents[idx].state_id < MAX_STATES) {
            atomicAdd(&g_stateCounts[d_incidents[idx].state_id], 1); // Atomic addition
        }
    }
}
