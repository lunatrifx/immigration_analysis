#include <iostream> 
#include <vector>
#include <map>
#include <string>
#include "data_loader.h"
#include "cuda_analyzer.h"
#include "visualization_manager.h" // for simple export

int main() {
    std::cout << "Starting Illegal Immigration Data Analysis Project" << std::endl;

    // Step 1: Load data
    DataLoader dataLoader;
    std::string dataFilePath = "C:\\Users\\pquin\\OneDrive\\Desktop\\immigration_analysis\\data\\Border_Crossing_Entry_Data.csv"; // Update with actual path
    std::vector<ImmigrationIncident> incidents = dataLoader.loadIncidents(dataFilePath);

    if (incidents.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return -1;
    }

    // Step 2: analyze data using CUDA
    std::cout << "Analyzing data using CUDA..." << std::endl;
    CUDAAnalyzer analyzer;
    std::map<std::string, int> StateIncidentCount = analyzer.analyzeIncidentsByState(incidents);

    // Step 3: Export results for visualization
    std::string outputCSV = "state_incident_counts.csv";
    VisualizationManager::exportStateCountsToCSV(StateIncidentCount, outputCSV);

    std::cout << "Version 1. Analysis complete. Please run 'visualization_data.py' to see the map." << std::endl;
    return 0;
}
