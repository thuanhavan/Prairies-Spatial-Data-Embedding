# Prairies-Spatial-Data-Embedding

Prairie Multi-Sensor Embedding Pipeline

This repository contains a complete workflow for generating multi-sensor geospatial embeddings for agricultural and environmental analysis across the Canadian Prairies. The workflow integrates optical, radar, climate, and topographic datasets into a unified 5 × 5 km grid system, followed by feature extraction using deep-learning autoencoders.

🌐 Data Sources
1. Sentinel-1 (SAR):VV/VH backscatter. 
Preprocessing: speckle filtering, orbit correction, clipping by tiles
Useful for soil moisture, surface structure, and crop/soil discrimination

2. Sentinel-2 (Optical MSI):RGB, NIR, SWIR, and Red-edge bands
Cloud masking and seasonal composite generation
Vegetation and moisture indices optionally included (NDVI, NDWI, etc.)

3. ERA5 Monthly Climate: Temperature (air temperature at 2 m)
Monthly precipitation
Resampled and aligned to the 5 × 5 km tile system
Represents climatic variability affecting crop and soil conditions

4. SRTM Topography: Elevation, Slope
Provides stable topographic predictors for hydrology, crop growth, and soil patterns

<img width="1390" height="695" alt="image" src="https://github.com/user-attachments/assets/b12f211b-8f98-44aa-9197-810d0096dbeb" />
