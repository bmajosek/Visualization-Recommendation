# Visualization Recommendation Project

This project uses Draco to recommend Vega-Lite visualizations based on input data. 

## Features

- **Column Selection**: An LLM selects two columns from the input DataFrame for visualization.
- **Chart Recommendation**: Generates Vega-Lite chart specifications using Draco.
- **Chart Evaluation**: Assigns scores to charts based on LLM evaluation, ensuring the best visualization quality.

## Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the notebook `visualize.ipynb` in any notebook editor.
2. Run all the cells in sequence.
3. The notebook will:
    - Load a sample DataFrame from vega_datasets.
    - Use an LLM to select columns and evaluate chart specifications.
    - Display recommended visualizations and their scores.