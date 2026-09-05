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

## 
The main difficulty I faced was finding the shortest path for the rover.

I tried using different path orders, but because of the large image size and the number of possible casualty permutations, the program was taking a very long time to complete.

Therefore, I used a simpler route-generation approach that visits the casualties and reaches the final destination while avoiding non-traversable areas. This allowed the program to process the images within a reasonable amount of time.