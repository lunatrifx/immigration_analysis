// Data structures

#pragma once 
#include <string>
#include <vector>
#include <map>
#include <set>

// Structure to hold one record from the dataset
struct ImmigrationIncident {
    std::string PortName;
    std::string State;
    std::string PortCode;
    std::string Border;
    std::string Date;
    std::string Measure;
    // Ommitting 'value' as unsure of representation
    double Latitude;
    double Longitude;
};

// data structure for analysis results (e,g, per state)
struct StateIncidentCount {
    std::string stateName;
    int count;
};

// In later versions, for a geospatial heatmap
struct GeoGridCell {
    double minLat, maxLat;
    double minLon, maxLon;
    int incidentCount;
};

// Function to declare for data loading
class DataLoader {
public:
    // Load data from CSV file
    static std::vector<ImmigrationIncident> loadDataFromCSV(const std::string& filePath);
};