#pragma once
#include <string>
#include <map>

class VisualizationManager {
    public:
       static void exportStateCountsToCSV(const std::map<std::string, int>& stateCounts, const std::string& filename);
       // Create a python script to visualize the CSV.
        

};

