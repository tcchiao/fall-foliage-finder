**Cindy Chiao**  
**April 5, 2016**  

# Final Project Proposal

### Predicting Timing of Fall Foliage (and Spring Leave-out event)

* Objective: The goal of this project is to predict the frontier of autumn leaves changing colors to enable better travel planning to enjoy the fall colors. 

* Data Pipeline: 
	* Input data: 
		* Satellite vegetation index (NVDI) - TSince this is a forecasting problem, the satellite data will be part of my input data and will also be used to calculate my labels (predict leave color at time (t+1)using data up to time t). For any given image, the pixel value will be classify as "green" vs. "non-green", with "non-green" signifying the period of time between Fall leaves changing color to Spring leave-out. Data is available in various resolution at 1 image per 8 days. The 1km x 1km product will be used for this project.  

		* Weather data (forecast and past data) - Precipitation, temperature, and amount of daylight. The data is likely in time series format for given locations, which will have to be interpolated into a grid format over the continental U.S. to match the satellite data.

		* Geographic information - elevation and slope. The data is available in raster format but may need to be re-project and re-sample to match the satellite data. 

	* Data Processing: 
		* A "row" of my data table will represent a specific time point at a given location within the domain. The input data will consist of: 1) satellite data up to time t at the given location and in vicinity of the location (50x50 box?) and 2) weather data (forecast and historic) at the given location, and 3) geographic information at the given location (static). 

	* Modeling: 
		* In general, the model(s) will be trained on past years with a few years reserved for validation. 
		* Multiple models can be trained on separate data sources then polled to generate final prediction (ensemble models)
		* Linear Regression, Neural Networks (satellite image), Random Regressor Forest, Boosting. 

	* (Optional) Ground Truth Labels: 
		* To validate satellite observations, ground truths can be assessed by pulling geotagged and timestamped Twitter feeds and/or Flickr images (i.e. Twitter feeds or images with hashtags related to fall foliage).

* Potential challenges: 
	* The first challenge will be to get all data formats into Python for any given location. This project will involve a significant amount of data munging and cleaning due to the number of different data sources. However, I have worked with weather and geographic data in the past and will only need to learn to work with the satellite data. 

	* Data shortage: For any given location in the U.S., there are only ~350 images available (~10 year/8 days). This number is too low to train separate models for each location. Thus, the model(s) built needs to be generalizable across different locations. This may be achieved by de-mean and standardized all input data (i.e. represent data as x stdev away from historic mean at that location). Alternative, locations similar in climate and geographic attributes can be cluster together using unsupervised algorithms (i.e. K-means) and different models can be built for each cluster. 

* Additional information source: 
	* A number of papers have explored the idea of using remote sensing data to measure plant phenology changes, with applications in understanding global change biology. These papers will be used to draw insights for the feature engineering and modeling phases of this project and will be cited accordingly. 

	* Visualization example: http://smokymountains.com/fall-foliage-map/ 
