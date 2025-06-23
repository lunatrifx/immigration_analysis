#include <iostream>
#include "cuda_analyzer.h"
#include <cuda_runtime.h>
#include <unordered_map> // for mapping state names to IDs on the host

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

   for (const auto& incident : incidents) {
    // Debug print
    std::cout << "Processing incident in state: " << incident.State << std::endl;
    // End debug print

    // Current place holder logic
    if (!incident.State.empty()) {
        hostStateCounts[incident.State]++;
    }
    else {
        hostStateCounts["UnknownBorder"]++;
    }
   }
}

// Creating the global array on the device to store state counts
__device__ int g_stateCounts[MAX_STATES];

// Initializing a CUDA kernel to count the incidents per state
__global__ void countIncidentsByStateKernel(DeviceIncident_GPU* d_incidents, int numIncidents) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numIncidents) {
        // This is assumning state_id is already mapped correctly on the host
        if (d_incidents[idx].state_id >= 0 && d_incidents[idx].state_id < MAX_STATES) {
            atomicAdd(&g_stateCounts[d_incidents[idx].state_id], 1); // Atomic addition
    }
 }
}

std::map<std::string, int> analyzeIncidentsByState(const std::vector<ImmigrationIncident>& incidents) {
    std::map<std::string, int> hostStateCounts;
    std::unordered_map<std::string, int> hostStateToID; // Host-side mapping of state names to IDs
    std::vector<std::string> idToState; // To convert back from ID to state name

    // 1. Preparing the data for transfer and creating the state ID on the host
    std::vector<DeviceIncident_GPU> h_incidents(incidents.size());
    int currentId = 0;
    for (size_t i = 0; i < incidents.size(); ++i) {
        h_incidents[i].Latitude = incidents[i].Latitude;
        h_incidents[i].Longitude = incidents[i].Longitude;
        h_incidents[i].border_type = (incidents[i].Border == "Canada") ? 0 : 1;

        // Map state name to an ID
        if (hostStateToID.find(incidents[i].State) == hostStateToID.end()) {
            hostStateToID[incidents[i].State] = currentId++;
            idToState.push_back(incidents[i].State);
            currentId++;
            if (currentId >= MAX_STATES) {
                std::cerr << "Exceeded maximum number of states supported." << std::endl;
                break; // Prevent overflow
            }
        }
        h_incidents[i].state_id = hostStateToID[incidents[i].State];
    }
    if (currentId >= MAX_STATES) {
        std::cerr << "Too many unique states in the input data." << std::endl;
        return hostStateCounts; // Return empty result
    }

    // 2. Allocating device memory
    DeviceIncident_GPU* d_incidents;
    CUDA_CHECK(cudaMalloc((void**)&d_incidents, h_incidents.size() * sizeof(DeviceIncident_GPU)));

    int* d_stateCounts;
    CUDA_CHECK(cudaMalloc((void**)&d_stateCounts, MAX_STATES * sizeof(int)));

    // Initialize device state counts to zero
    CUDA_CHECK(cudaMemset(d_stateCounts, 0, MAX_STATES * sizeof(int)));

    // 3. Copying data to device
    CUDA_CHECK(cudaMemcpy(d_incidents, h_incidents.data(), h_incidents.size() * sizeof(DeviceIncident_GPU), cudaMemcpyHostToDevice));

    // For the global array (g_stateCounts) which is on the device, we need to declare it
    // within the .cu file or use cudaGetSymbolAddress or cudaMemset if it's external
    // For version 1, using a local device array passed as a parameter is safer and more flexible.
    // Let's modify the kernel to use d_stateCounts directly.

    // 4. Launch Kernel
    // Change when confident about performance, understand CUDA more
    int threadsPerBlock = 256;
    int blocksPerGrid = (h_incidents.size() + threadsPerBlock - 1) / threadsPerBlock;
    countIncidentsByStateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_incidents, h_incidents.size());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for the kernel to complete

    // 5. Send the results back to the host
    std::vector<int> h_stateCounts(currentId);
    CUDA_CHECK(cudaMemcpy(h_stateCounts.data(), d_stateCounts, currentId * sizeof(int), cudaMemcpyDeviceToHost));

    // 6. Free device memory
    CUDA_CHECK(cudaFree(d_incidents));
    CUDA_CHECK(cudaFree(d_stateCounts));

    // 7. Populate the host map with results
    for (int i = 0; i < currentId; ++i) {
        hostStateCounts[idToState[i]] = h_stateCounts[i];
    }
    return hostStateCounts;

}
