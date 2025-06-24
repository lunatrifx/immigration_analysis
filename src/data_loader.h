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
    int Date;
    int Measure;
    int value;
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
    std::vector<ImmigrationIncident> loadIncidents(const std::string& filename);
};