#include <iostream>
#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <limits> // For std::numeric_limits
#include <algorithm> // For std::remove_if, std::isspace
#include <cctype> // For std::isspace


// Creating a function to trim whitespace and trailing whitepace
static inline std::string& ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(),[](unsigned char ch) {
        return !std::isspace(ch);
    }));
    return s;
}

static inline std::string& rtrim (std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
    return s;
}

// Trim from both ends
static inline std::string& trim(std::string& s) {
    return ltrim(rtrim(s));
}
//end trimming function 

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

        if(!std::getline(ss, incident.Border, ',')) {
            std::cerr << "Error Parsing Border from line: " << line << std::endl;
        }
        trim(incident.Border); // Trim whitespace from Border

        // Debug Prints
        std::cout << "Parsed Incident: PortName=' " << incident.PortName << " ' "
                << "', Border=' " << incident.Border << " ' "
                << "', Latitude=" << incident.Latitude << " ' "
                << ", Longitude=" << incident.Longitude << " ' "
                << std::endl;

                // end Debug Prints

        incidents.push_back(incident);
    }
    file.close();
    std::cout << "Loaded " << incidents.size() << " records from " << filePath << std::endl;
    return incidents;

}
