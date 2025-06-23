#include "visualization_manager.h"
#include <fstream>
#include <iostream>

void VisualizationManager::exportStateCountsToCSV(const std::map<std::string, int>& stateCounts, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    // Write CSV header
    outfile << "State,IncidentCount\n";
    for (const auto& pair : stateCounts) {
        outfile << pair.first << "," << pair.second << "\n";
    }

    outfile.close();
    std::cout << "State counts exported to " << filename << std::endl;
}
