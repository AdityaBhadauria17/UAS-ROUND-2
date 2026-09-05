import os
import subprocess
import re
import shutil


INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FIND ALL IMAGES
# ============================================================

image_files = []

for filename in os.listdir(INPUT_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_files.append(filename)

image_files.sort()


print()
print("========================================")
print("UAS-DTU BATCH PROCESSING")
print("========================================")
print("Images found:", len(image_files))


# ============================================================
# STORE RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for filename in image_files:

    print()
    print("----------------------------------------")
    print("Processing:", filename)
    print("----------------------------------------")

    image_path = os.path.join(INPUT_FOLDER, filename)

    process = subprocess.run(
        ["python", "main.py", image_path],
        capture_output=True,
        text=True
    )

    output = process.stdout
    error = process.stderr

    print(output)

    if error:
        print("ERROR:")
        print(error)

    # --------------------------------------------------------
    # GET TOTAL PATH POINTS
    # --------------------------------------------------------

    path_match = re.search(
        r"Total Path Points:\s*([0-9]+)",
        output
    )

    if path_match:
        path_points = int(path_match.group(1))
    else:
        path_points = 0

    # --------------------------------------------------------
    # GET TRAVEL TIME
    # --------------------------------------------------------

    time_match = re.search(
        r"TOTAL TRAVEL TIME:\s*([0-9.]+)",
        output
    )

    if time_match:
        travel_time = float(time_match.group(1))
    else:
        travel_time = 0.0

    # --------------------------------------------------------
    # CREATE UNIQUE OUTPUT NAME
    # --------------------------------------------------------

    base_name = os.path.splitext(filename)[0]

    source_route = os.path.join(
        OUTPUT_FOLDER,
        "optimized_rover_route.png"
    )

    unique_route = os.path.join(
        OUTPUT_FOLDER,
        "optimized_rover_route_" + base_name + ".png"
    )

    if os.path.exists(source_route):

        shutil.copy2(
            source_route,
            unique_route
        )

        print(
            "Route copied to:",
            unique_route
        )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "image": filename,
            "path_points": path_points,
            "time": travel_time
        }
    )


# ============================================================
# PATH RANKING
# ============================================================

path_ranking = sorted(
    results,
    key=lambda x: x["path_points"]
)


# ============================================================
# TIME RANKING
# ============================================================

time_ranking = sorted(
    results,
    key=lambda x: x["time"]
)


# ============================================================
# PRINT PATH RANKING
# ============================================================

print()
print("========================================")
print("GLOBAL PATH RANKING")
print("========================================")

for rank, result in enumerate(
    path_ranking,
    start=1
):

    print(
        rank,
        "->",
        result["image"],
        "| Path Points:",
        result["path_points"]
    )


# ============================================================
# PRINT TIME RANKING
# ============================================================

print()
print("========================================")
print("GLOBAL TIME RANKING")
print("========================================")

for rank, result in enumerate(
    time_ranking,
    start=1
):

    print(
        rank,
        "->",
        result["image"],
        "| Time:",
        round(result["time"], 4),
        "seconds"
    )


# ============================================================
# SAVE GLOBAL RANKING
# ============================================================

ranking_file = os.path.join(
    OUTPUT_FOLDER,
    "global_ranking.txt"
)

with open(
    ranking_file,
    "w",
    encoding="utf-8"
) as file:

    file.write("UAS-DTU GLOBAL RANKING\n")
    file.write("======================\n\n")

    file.write("PATH RANKING\n")
    file.write("------------\n")

    for rank, result in enumerate(
        path_ranking,
        start=1
    ):

        file.write(
            f"{rank}. "
            f"{result['image']} - "
            f"Path Points = "
            f"{result['path_points']}\n"
        )

    file.write("\n")

    file.write("TIME RANKING\n")
    file.write("------------\n")

    for rank, result in enumerate(
        time_ranking,
        start=1
    ):

        file.write(
            f"{rank}. "
            f"{result['image']} - "
            f"Time = "
            f"{result['time']:.4f} seconds\n"
        )


print()
print(
    "Global ranking saved at:",
    ranking_file
)

print()
print("========================================")
print("BATCH PROCESSING COMPLETE")
print("========================================")