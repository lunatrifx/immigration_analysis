#include <iostream>
#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <limits> // For std::numeric_limits
#include <algorithm> // For std::remove_if, std::isspace
#include <cctype> // For std::isspace
#include <sstream> // For std::istringstream


// Creating a function to trim whitespace and trailing whitepace
static inline std::string& ltrim(std::string& s) {/*...*/ return s;}
static inline std::string& rtrim(std::string& s) {/*...*/ return s;}
static inline std::string& trim (std::string&s) {/*...*/ return ltrim(rtrim(s));}
//end trimming function 

std::vector<ImmigrationIncident> DataLoader::loadIncidents(const std::string& filename) {
    std::vector<ImmigrationIncident> incidents;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return incidents;
    }

    std::string line;
    // Read and discard header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        ImmigrationIncident incident;

        // --- NEW PARSING SEQUENCE - MUST MATCH CSV HEADER EXACTLY ---
        // Port Name
        if (!std::getline(iss, incident.PortName, ',')) { std::cerr << "Error parsing Port Name" << std::endl; continue; }
        trim(incident.PortName); // Trim any spaces

        // State
        if (!std::getline(iss, incident.State, ',')) { std::cerr << "Error parsing State" << std::endl; continue; }
        trim(incident.State); // Trim any spaces

        // Port Code
        if (!std::getline(iss, incident.PortCode, ',')) { std::cerr << "Error parsing Port Code" << std::endl; continue; }
        trim(incident.PortCode);

        // Border
        if (!std::getline(iss, incident.Border, ',')) { std::cerr << "Error parsing Border" << std::endl; continue; }
        trim(incident.Border); // This is where the 'US-Canada Border' will now be read!

        // Date (assuming int conversion)
        if (std::getline(iss, token, ',')) {
            try { incident.Date = std::stoi(token); } catch(...) { incident.Date = 0; std::cerr << "Error converting Date: " << token << std::endl; }
        } else { std::cerr << "Error parsing Date" << std::endl; continue; }

        // Measure (assuming int conversion)
        if (std::getline(iss, token, ',')) {
            try { incident.Measure = std::stoi(token); } catch(...) { incident.Measure = 0; std::cerr << "Error converting Measure: " << token << std::endl; }
        } else { std::cerr << "Error parsing Measure" << std::endl; continue; }

        // Value (assuming int conversion)
        if (std::getline(iss, token, ',')) {
            try { incident.value = std::stoi(token); } catch(...) { incident.value = 0; std::cerr << "Error converting Value: " << token << std::endl; }
        } else { std::cerr << "Error parsing Value" << std::endl; continue; }

        // Latitude (assuming double conversion)
        if (std::getline(iss, token, ',')) {
            try { incident.Latitude = std::stod(token); } catch(...) { incident.Latitude = 0.0; std::cerr << "Error converting Latitude: " << token << std::endl; }
        } else { std::cerr << "Error parsing Latitude" << std::endl; continue; }

        // Longitude (assuming double conversion, and it's the second to last)
        if (std::getline(iss, token, ',')) {
            try { incident.Longitude = std::stod(token); } catch(...) { incident.Longitude = 0.0; std::cerr << "Error converting Longitude: " << token << std::endl; }
        } else { std::cerr << "Error parsing Longitude" << std::endl; continue; }

        // Point (last column, read until end of line, then discard)
        std::getline(iss, token); // Reads the rest of the line (Point column)

        // --- CORRECTED DEBUG PRINTS ---
        std::cout << "Parsed Incident: PortName='" << incident.PortName << "'"
                  << ", State='" << incident.State << "'" // Now print the State
                  << ", Border='" << incident.Border << "'" // This should be correct now!
                  << ", Latitude=" << incident.Latitude
                  << ", Longitude=" << incident.Longitude // This should now be correct!
                  << std::endl;

        incidents.push_back(incident);
    }
    return incidents;
}
