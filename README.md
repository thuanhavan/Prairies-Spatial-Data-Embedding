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


🗂️ Spatial Framework
5 × 5 km Grid System
Region covered: Prairie Provinces (SK, AB, MB)
133 total tiles (testing)
Each tile is processed independently to support parallelization and cloud workflows
Ensures consistent alignment across Sentinel-1, Sentinel-2, ERA5, and SRTM datasets

🤖 Embedding Generation (Autoencoder)
A convolutional autoencoder model is trained to compress multi-sensor inputs into compact feature vectors (embeddings). These embeddings capture spectral, structural, climatic, and topographic signatures of the landscape.
Autoencoder Workflow
Preprocess Sentinel-1, Sentinel-2, ERA5, and SRTM inputs
Stack features into multi-band arrays per 5 × 5 km tile
Train autoencoder to learn a representation of the full multi-sensor stack
Extract embedding vectors for each pixel or patch
Save embeddings for downstream machine learning tasks

<img width="1390" height="695" alt="image" src="https://github.com/user-attachments/assets/b12f211b-8f98-44aa-9197-810d0096dbeb" />

🤖  Applications:
Crop type classification
Field boundary extraction
Soil moisture and soil EC prediction
Water occurrence modeling
Change detection
General geospatial clustering and feature learning


🚀 Goals of This Project
Build a unified geospatial feature representation for Prairie environments
Enable fast ML model training on compressed features instead of raw imagery
Support scalable mapping tasks using embeddings (classification, prediction, monitoring)
Provide a modular pipeline for integration into Google Earth Engine or HPC workflows

📬 Contact
Thuan Ha
Research Officer, University of Saskatchewan
Email: thuan.ha@usask.ca
GitHub: https://github.com/thuanhavan
