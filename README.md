# Injury Prediction Project

## Overview

This project aims to predict injuries in athletes using machine learning techniques. The project involves data preprocessing, feature selection, and model training using decision trees and neural networks. The data is collected from athletes' training sessions and includes various metrics such as total kilometers run, perceived exertion, and recovery.

## Project Structure

- **data/**: Contains the raw data files used for training and testing the models.
  - `week_approach_maskedID_timeseries.csv`: Weekly aggregated data.
  - `day_approach_maskedID_timeseries.csv`: Daily data.

- **processing/**: Contains scripts for data preprocessing.
  - `preprocessing.py`: Preprocesses the raw data and saves it for further analysis.

- **analysis/**: Contains results of feature selection and model outputs.
  - `weekly/selected_features_nn.csv`: Selected features for the neural network model.
  - `daily/selected_features_lasso.csv`: Selected features using Lasso regression.

- **clustering/**: Contains scripts for clustering analysis.
  - `flock_clustering.py`: Implements a flocking model to simulate and analyze the movement patterns of athletes.

## Flock Clustering

The flock clustering component of the project simulates the movement patterns of athletes using an agent-based model inspired by Craig Reynolds' Boids model. This model is used to understand how athletes' movement patterns can be grouped and analyzed for potential injury prediction.

### Key Features

- **Boid Agents**: Each athlete is represented as a "boid" agent with a position and velocity in a continuous space.
- **Flocking Rules**: The boids follow four main rules:
  1. **Cohesion**: Boids steer towards the average position of their neighbors.
  2. **Separation**: Boids steer to avoid crowding local flockmates.
  3. **Alignment**: Boids steer towards the average heading of their neighbors.
  4. **Borders**: Boids avoid the edges of the simulation space.
- **Simulation**: The model simulates the movement of boids over a specified number of steps, and the final positions are saved for analysis.

### Running the Flock Clustering

To run the flock clustering simulation, execute the following command:

```bash
python clustering/flock_clustering.py
```

This will generate an animation of the boids' movement and save it as `boids_2.gif`.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Carzgil/running_prediction
   cd running_injury
   ```

2. **Install dependencies**:
   Ensure you have Python 3.x installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess the data**:
   Run the preprocessing script to prepare the data:
   ```bash
   python processing/preprocessing.py
   ```

4. **Train the models**:
   - **Decision Tree**: Run the decision tree script:
     ```bash
     python decision_tree.py
     ```

   - **Neural Network**: Run the neural network script:
     ```bash
     python feature_nn.py
     ```

### Results

- The decision tree visualization can be found in the `analysis` folder.
- Selected features and trained models are saved in their respective directories.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.


