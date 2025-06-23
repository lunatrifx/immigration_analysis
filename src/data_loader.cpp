#include <iostream>
#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <limits> // For std::numeric_limits

std::vector<ImmigrationIncident> DataLoader::loadDataFromCSV(const std::string& filePath) {
    std::vector<ImmigrationIncident> incidents;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return incidents;
    }

    std::string line;
    // Skip header line
    std::getline(file, line);

    // Parsing CSV line by line
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        ImmigrationIncident incident;

        // Assuming CSV columns are in the order of the struct fields
        std::getline(ss, incident.PortName, ',');
        std::getline(ss, incident.State, ',');
        std::getline(ss, incident.PortCode, ',');
        std::getline(ss, incident.Border, ',');
        std::getline(ss, incident.Date, ',');
        std::getline(ss, incident.Measure, ',');

        // Skip 'value' field as representation is unclear
        std::getline(ss, token, ','); // Skip value

        // Latitude
        std::getline(ss, token, ',');
        try {
            incident.Latitude = std::stod(token);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Could not locate latitude ' " << token << " ' . Error: " << e.what() << std::endl;
             incident.Latitude = 0.0; // Default or error value

        }

        // Debug Prints
        std::cout << "Parsed Incident: PortName=' " << incident.PortName << " ' "
                << "', Longitude=' " << incident.Border << " ' "
                << "', Lat=" << incident.Latitude << " ' "
                << ", Lon=" << incident.Longitude << " ' "
                << std::endl;

                // end Debug Prints

        incidents.push_back(incident);
    }
    file.close();
    std::cout << "Loaded " << incidents.size() << " records from " << filePath << std::endl;
    return incidents;

}
