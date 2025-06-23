# Illegal Immigration Data Analysis

This project prototypes a C++ and CUDA-based analysis of a Department of Defense dataset on illegal immigration incidents/apprehensions, aiming to visualize actual incident locations and qualitatively compare them against perceived media biases in an external thesis.

## Dataset

Expected data format: CSV file with 11 columns. The relevant columns for this project are:
`Port, State, Port Code, Border (US-Canada, US-Mexico), date, Measure, value (OMITTED), Latitude, Longitude`

## Project Structure

* `src/`: Contains all C++ and CUDA source files.
* `data/`: Placeholder for your `dod_immigration_data.csv` file.
* `CMakeLists.txt`: CMake build configuration.
* `visualize_data.py`: Python script for generating the map visualization.

## Build Instructions

1.  **Prerequisites:**
    * CUDA Toolkit (compatible with your NVIDIA GPU and Visual Studio/GCC)
    * C++ Compiler (GCC/Clang for Linux/macOS, MSVC for Windows)
    * CMake (version 3.10+)
    * Python 3.x with `pandas`, `geopandas`, `matplotlib`. Install with pip:
        `pip install pandas geopandas matplotlib`
    * You'll need a shapefile for country/state boundaries for `geopandas`. A good source is Natural Earth Data (e.g., `natural_earth_lowres.zip` for countries). Place it where your Python script can find it, or update the script to download it.

2.  **Place your dataset:**
    * Save your declassified DoD dataset as `dod_immigration_data.csv` inside the `data/` directory.

3.  **Build the C++ Application:**

    ```bash
    mkdir build
    cd build
    cmake ..
    make # Or `cmake --build .` on Windows/Visual Studio
    ```

    This will create an executable (e.g., `ImmigrationAnalysis`) in `build/bin/`.

## Running the Analysis

1.  **Execute the C++ program:**

    ```bash
    ./bin/ImmigrationAnalysis
    ```

    This will:
    * Load your data.
    * Process it using CUDA to count incidents per state.
    * Generate a `state_incident_counts.csv` file in the `build/bin/` directory.

2.  **Visualize the results:**
    * Ensure you have the necessary Python libraries and a state shapefile.
    * From the `build/bin/` directory (or wherever `state_incident_counts.csv` is generated):

    ```bash
    python ../../visualize_data.py
    ```
    This script will generate a choropleth map showing the distribution of incidents by state.

## Testing the Thesis (Media Bias)

After the map is generated, visually compare the "hotspots" on your map (where actual illegal immigration incidents are highest) with your understanding or anecdotal evidence of where the media primarily reports on illegal immigration.

* **Question:** Do the areas with the most incidents align with the areas most frequently highlighted by news media regarding illegal immigration?
* **Hypothesis Validation:** If the areas *do not* align, it supports your thesis of media bias. If they largely align, it suggests less bias in location reporting.

**Future Enhancements for a Deeper Analysis:**

* **Quantitative Media Analysis:** Integrate a dataset of media mentions, potentially by scraping news articles and using NLP to extract locations and then geocoding them. This would allow for a data-driven comparison.
* **Time Series Analysis:** Analyze trends over time (e.g., month by month) on both the incident data and media data.
* **More Sophisticated CUDA Analysis:** Implement geospatial binning to create a heatmap instead of just state-level counts, providing finer granularity.
* **Interactive C++ UI:** Use Qt for a fully integrated desktop application with an interactive map, allowing zooming, panning, and filtering.
