# UAS-DTU Round 2 - Rover Casualty Analysis

## Project Overview

I developed this project for the UAS-DTU Round 2 software task.

The program processes aerial images and helps the rover identify casualties, analyse the terrain and generate a safe route from the starting point to the final destination.

## Features

- Reads aerial images using OpenCV
- Converts images from BGR to HSV
- Creates a traversability mask
- Detects non-traversable areas such as black obstacles and blue water
- Detects casualties from the input images
- Identifies casualty coordinates
- Classifies casualty shape, colour and age group
- Calculates casualty severity and priority score
- Finds a rover route through the casualties
- Calculates rover travel time based on terrain
- Generates mask images
- Generates optimized rover route images
- Supports batch processing of multiple images
- Generates a global ranking based on path score and travel time

## Technologies Used

- Python
- OpenCV
- NumPy
- Heapq
- Math
- Regular Expressions

## Project Structure

text
UAS-ROUND-2/
│
├── input/
│    Input aerial images
│
├── output/
│   Traversability masks
│   Optimized rover route images
│   Global ranking
│
├── src/
│
├── main.py
├── batch_runner.py
└── README.md

## How to Run

### 1. Install required libraries

```bash
pip install opencv-python numpy
python batch_runner.py (to run multiple input images)
python main.py input/IMG-20260831-WA0025.jpg (to run single image)

# THE PART OF THE CODE I WAS NOT ABLE TO DO WAS TO FIND THE SHORTEST PATH BECOZ AFTER TRYING MANY TIMES AND DUE TO LARGE PIXELS AND NUMBER OF PERMUTATIONS THE CODE WAS TAKING HOURS TO COMPLETE THAT ALTHOUGH I TRIED BUT I WAS NOT ABLE TO THINK ANY OTHER METHOD SO INSTEAD I ADDED ONLY PATH WITH NO CONDITION THAT EXECUTES IN SHORT TIME