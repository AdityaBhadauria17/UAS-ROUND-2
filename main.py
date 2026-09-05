import cv2
import numpy as np
import heapq
import math
import itertools
import os
import sys

# STEP 1: LOAD IMAGE

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "input/IMG-20260831-WA0025.jpg"

image = cv2.imread(image_path)

if image is None:
    print("Image could not be loaded.")
    exit()

print("Image loaded successfully.")
print("Image shape:", image.shape)

os.makedirs("output", exist_ok=True)

# ============================================================
# STEP 2: CONVERT IMAGE TO HSV

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("HSV conversion completed.")

# ============================================================
# STEP 3: CREATE TRAVERSABILITY MASK

# ------------------------------------------------------------
# BLACK OBSTACLES

lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 80, 70])

black_mask = cv2.inRange(
    hsv,
    lower_black,
    upper_black
)
kernel = np.ones((15, 15), np.uint8)

black_mask = cv2.dilate(
    black_mask,
    kernel,
    iterations=1
)
# ------------------------------------
# BLUE WATER

lower_blue = np.array([90, 80, 40])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(
    hsv,
    lower_blue,
    upper_blue
)
# ------------------------------------------------------------
# COMBINE
non_traversable_mask = cv2.bitwise_or(
    black_mask,
    blue_mask
)
# ------------------------------------------------------------
# REMOVE VERY SMALL NOISE

kernel = np.ones((3, 3), np.uint8)

non_traversable_mask = cv2.morphologyEx(
    non_traversable_mask,
    cv2.MORPH_OPEN,
    kernel
)

# ------------------------------------------------------------
# TRAVERSABILITY

# White = traversable
# Black = non-traversable

traversability_mask = cv2.bitwise_not(
    non_traversable_mask
)

# ============================================================
# SAFETY MARGIN AROUND OBSTACLES

kernel_safe = np.ones((15, 15), np.uint8)

safe_traversability_mask = cv2.erode(
    traversability_mask,
    kernel_safe,
    iterations=1
)

# Saving mask

mask_output_path = "output/mask_" + os.path.basename(image_path)
mask_output_path = os.path.splitext(mask_output_path)[0] + ".png"

cv2.imwrite(
    mask_output_path,
    traversability_mask
)

print("Traversability mask saved.")
print("Saved at:", mask_output_path)

# ============================================================
# STEP 4: DETECT CASUALTIES

# ------------------------------------------------------------
# RED

lower_red1 = np.array([0, 80, 60])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 80, 60])
upper_red2 = np.array([180, 255, 255])

red_mask1 = cv2.inRange(
    hsv,
    lower_red1,
    upper_red1
)

red_mask2 = cv2.inRange(
    hsv,
    lower_red2,
    upper_red2
)

red_mask = cv2.bitwise_or(
    red_mask1,
    red_mask2
)
# ------------------------------------------------------------
# YELLOW

lower_yellow = np.array([15, 80, 80])
upper_yellow = np.array([40, 255, 255])

yellow_mask = cv2.inRange(
    hsv,
    lower_yellow,
    upper_yellow
)
# ------------------------------------------------------------
# WHITE

lower_white = np.array([0, 0, 180])
upper_white = np.array([180, 80, 255])

white_mask = cv2.inRange(
    hsv,
    lower_white,
    upper_white
)

# Combining casualty masks

casualty_mask = cv2.bitwise_or(
    red_mask,
    yellow_mask
)

casualty_mask = cv2.bitwise_or(
    casualty_mask,
    white_mask
)
# Remove small noise

casualty_mask = cv2.morphologyEx(
    casualty_mask,
    cv2.MORPH_OPEN,
    kernel
)

# Finding contours

contours, _ = cv2.findContours(
    casualty_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

casualty_detection_image = image.copy()

casualties = []

# ============================================================
# PROCESS EACH CASUALTY

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 50:
        continue

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter == 0:
        continue

    # --------------------------------------------------------
    # CENTRE
   
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        continue

    x = int(
        moments["m10"] / moments["m00"]
    )

    y = int(
        moments["m01"] / moments["m00"]
    )

    # --------------------------------------------------------
    # SHAPE
    approx = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )
    vertices = len(approx)

    circularity = (
        4 * np.pi * area
        / (perimeter * perimeter)
    )
    if vertices == 4:

        shape = "Square"
        age_group = "Senior Citizen"
        age_score = 2

    elif circularity > 0.70 and vertices >= 8:

        shape = "Circle"
        age_group = "Child"
        age_score = 3

    elif vertices >= 5:

        shape = "Star"
        age_group = "Adult"
        age_score = 1

    else:

        shape = "Unknown"
        age_group = "Unknown"
        age_score = 0

    # --------------------------------------------------------
    # COLOUR
    contour_mask = np.zeros_like(casualty_mask)

    cv2.drawContours(
        contour_mask,
        [contour],
        -1,
        255,
        -1
    )

    mean_hsv = cv2.mean(
        hsv,
        mask=contour_mask
    )

    hue = mean_hsv[0]
    saturation = mean_hsv[1]
    value = mean_hsv[2]


      # --------------------------------------------------------
    # COLOUR AND SEVERITY

    red_pixels = np.sum(
        red_mask[contour_mask > 0] > 0
    )

    yellow_pixels = np.sum(
        yellow_mask[contour_mask > 0] > 0
    )

    white_pixels = np.sum(
        white_mask[contour_mask > 0] > 0
    )

    if red_pixels >= yellow_pixels and red_pixels >= white_pixels:

        colour = "Red"
        severity = "Critical"
        severity_score = 3

    elif yellow_pixels >= red_pixels and yellow_pixels >= white_pixels:

        colour = "Yellow"
        severity = "Moderate"
        severity_score = 2

    elif white_pixels >= red_pixels and white_pixels >= yellow_pixels:

        colour = "White"
        severity = "Safe"
        severity_score = 1

    else:

        colour = "Unknown"
        severity = "Unknown"
        severity_score = 0

    # --------------------------------------------------------
    # PRIORITY SCORE

    priority_score = (
        age_score * severity_score
    )

    # --------------------------------------------------------
    # STORE INFORMATION
    casualty = {

        "coordinate": (x, y),
        "shape": shape,
        "age_group": age_group,
        "age_score": age_score,
        "colour": colour,
        "severity": severity,
        "severity_score": severity_score,
        "priority_score": priority_score
    }

    casualties.append(casualty)

    # --------------------------------------------------------
    # DRAW CASUALTY DETECTION
    cv2.drawContours(
        casualty_detection_image,
        [contour],
        -1,
        (255, 0, 255),
        2
    )

    cv2.circle(
        casualty_detection_image,
        (x, y),
        5,
        (255, 0, 255),
        -1
    )

    label = f"{shape} - {colour}"

    cv2.putText(
        casualty_detection_image,
        label,
        (x - 50, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1
    )
# ============================================================
# STEP 5: PRINT CASUALTY INFORMATION

print()
print("Casualty Information")
print("====================")

for i, casualty in enumerate(casualties, start=1):

    print()
    print("Casualty", i)

    print(
        "Coordinate:",
        casualty["coordinate"]
    )

    print(
        "Shape:",
        casualty["shape"]
    )

    print(
        "Age Group:",
        casualty["age_group"]
    )

    print(
        "Age Score:",
        casualty["age_score"]
    )

    print(
        "Colour:",
        casualty["colour"]
    )

    print(
        "Severity:",
        casualty["severity"]
    )

    print(
        "Severity Score:",
        casualty["severity_score"]
    )

    print(
        "Priority Score:",
        casualty["priority_score"]
    )


print()
print(
    "Total casualties detected:",
    len(casualties)
)

# ============================================================
# STEP 6: ELEVATION LEVEL
def get_elevation_level(x, y):

    radius = 20

    x1 = max(0, x - radius)
    x2 = min(hsv.shape[1], x + radius)

    y1 = max(0, y - radius)
    y2 = min(hsv.shape[0], y + radius)

    area = hsv[y1:y2, x1:x2]

    # Light green

    light_lower = np.array([40, 80, 120])
    light_upper = np.array([80, 255, 255])

    light_mask = cv2.inRange(
        area,
        light_lower,
        light_upper
    )

    # Medium green

    medium_lower = np.array([40, 80, 60])
    medium_upper = np.array([80, 255, 180])

    medium_mask = cv2.inRange(
        area,
        medium_lower,
        medium_upper
    )

    # Dark green

    dark_lower = np.array([40, 80, 0])
    dark_upper = np.array([80, 255, 100])

    dark_mask = cv2.inRange(
        area,
        dark_lower,
        dark_upper
    )

    light_count = np.sum(
        light_mask > 0
    )

    medium_count = np.sum(
        medium_mask > 0
    )

    dark_count = np.sum(
        dark_mask > 0
    )

    if (
        light_count >= medium_count
        and light_count >= dark_count
    ):

        return 0

    elif (
        medium_count >= light_count
        and medium_count >= dark_count
    ):

        return 1

    else:

        return 2

print()
print("Casualty Elevation Information")
print("==============================")

for i, casualty in enumerate(
    casualties,
    start=1
):

    x, y = casualty["coordinate"]

    elevation = get_elevation_level(
        x,
        y
    )

    casualty["elevation"] = elevation

    print()
    print("Casualty", i)
    print(
        "Coordinate:",
        casualty["coordinate"]
    )

    print(
        "Elevation Level:",
        elevation
    )

# ============================================================
# STEP 7: DETECT ROVER START

lower_orange = np.array([5, 80, 80])
upper_orange = np.array([25, 255, 255])

orange_mask = cv2.inRange(
    hsv,
    lower_orange,
    upper_orange
)

orange_contours, _ = cv2.findContours(
    orange_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

start_point = None

if orange_contours:

    largest_orange = max(
        orange_contours,
        key=cv2.contourArea
    )

    M = cv2.moments(
        largest_orange
    )

    if M["m00"] != 0:

        start_x = int(
            M["m10"] / M["m00"]
        )

        start_y = int(
            M["m01"] / M["m00"]
        )

        start_point = (
            start_x,
            start_y
        )

# ============================================================
# STEP 8: DETECT FINAL DESTINATION

lower_purple = np.array([125, 50, 50])
upper_purple = np.array([170, 255, 255])

purple_mask = cv2.inRange(
    hsv,
    lower_purple,
    upper_purple
)

purple_contours, _ = cv2.findContours(
    purple_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

end_point = None

if purple_contours:

    # Finding purple triangle only
    triangle_contours = []

    for contour in purple_contours:

        area = cv2.contourArea(contour)

        if area < 500:
            continue

        perimeter = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        # Triangle has 3 corner points
        if len(approx) == 3:
            triangle_contours.append(contour)

    if triangle_contours:

        # Selecting the largest triangular contour
        triangle = max(
            triangle_contours,
            key=cv2.contourArea
        )

        M = cv2.moments(triangle)

        if M["m00"] != 0:

            end_x = int(
                M["m10"] / M["m00"]
            )

            end_y = int(
                M["m01"] / M["m00"]
            )

            end_point = (
                end_x,
                end_y
            )

print()
print("Rover Start and Destination")
print("===========================")

print(
    "Start Point:",
    start_point
)

print(
    "Final Destination:",
    end_point
)

# ===============================================
# CHECK POINTS
if start_point is None:

    print("ERROR: Rover start point not detected.")
    exit()

if end_point is None:

    print("ERROR: Final destination not detected.")
    exit()

if len(casualties) == 0:

    print("ERROR: No casualties detected.")
    exit()

# ================================================
# STEP 9: A* SAFE PATHFINDING

def heuristic(a, b):

    # Manhattan distance
    # Only Up / Down / Left / Right movement

    return (
        abs(a[0] - b[0])
        +
        abs(a[1] - b[1])
    )

# ============================================================
# SAFE A* PATH FINDING

def find_safe_path(mask, start, goal):

    height, width = mask.shape

    # Convert coordinates to integers
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    # --------------------------------------------------------
    # FIND NEAREST SAFE PIXEL
    # --------------------------------------------------------

    def nearest_safe_point(point):

        px, py = point

        # If point itself is safe
        if (
            0 <= px < width
            and 0 <= py < height
            and mask[py, px] > 0
        ):
            return point

        # Search nearby pixels
        for radius in range(1, 501):

            for dx in range(-radius, radius + 1):

                for dy in range(-radius, radius + 1):

                    nx = px + dx
                    ny = py + dy

                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                    ):

                        if mask[ny, nx] > 0:
                            return (nx, ny)

        return None

    # -------------------------------------
    # FIX START POINT

    safe_start = nearest_safe_point(start)

    if safe_start is None:

        print(
            "Could not find safe pixel near START:",
            start
        )

        return []

    # --------------------------------------------------------
    # FIX GOAL POINT

    safe_goal = nearest_safe_point(goal)

    if safe_goal is None:

        print(
            "Could not find safe pixel near GOAL:",
            goal
        )

        return []

    # Printing adjustment information
    if safe_start != start:

        print(
            "Start adjusted:",
            start,
            "->",
            safe_start
        )

    if safe_goal != goal:

        print(
            "Goal adjusted:",
            goal,
            "->",
            safe_goal
        )

    # --------------------------------------------------------
    # A* MOVEMENT DIRECTIONS
    
    directions = [

        (0, -1),   # UP
        (0, 1),    # DOWN
        (-1, 0),   # LEFT
        (1, 0),    # RIGHT

        (-1, -1),  # UP-LEFT
        (1, -1),   # UP-RIGHT
        (-1, 1),   # DOWN-LEFT
        (1, 1)     # DOWN-RIGHT
    ]

    # --------------------------------------------------------
    # HEURISTIC
    def distance(a, b):

        return math.sqrt(
            (a[0] - b[0]) ** 2
            +
            (a[1] - b[1]) ** 2
        )

    # --------------------------------------------------------
    # OPEN LIST

    open_list = []

    heapq.heappush(
        open_list,
        (
            distance(safe_start, safe_goal),
            0,
            safe_start
        )
    )

    # --------------------------------------------------------
    # A* DATA
    
    came_from = {}

    cost_so_far = {}

    came_from[safe_start] = None

    cost_so_far[safe_start] = 0

    # --------------------------------------------------------
    # A* SEARCH
    
    while open_list:

        priority, current_cost, current = heapq.heappop(
            open_list
        )

        # ----------------------------------------------------
        # GOAL REACHED
        
        if current == safe_goal:

            path = []

            while current is not None:

                path.append(current)

                current = came_from[current]

            path.reverse()

            return path

        current_x = current[0]
        current_y = current[1]

        # ----------------------------------------------------
        # CHECK NEIGHBOURS
        for dx, dy in directions:

            next_x = current_x + dx
            next_y = current_y + dy

            # Image boundary check

            if next_x < 0 or next_x >= width:
                continue

            if next_y < 0 or next_y >= height:
                continue

            # ------------------------------------------------
            # TRAVERSABILITY CHECK
            
            if mask[next_y, next_x] == 0:

                
                if (next_x, next_y) != safe_goal:
                    continue

            next_point = (next_x, next_y)

            # ------------------------------------------------
            # MOVEMENT COST
            if dx != 0 and dy != 0:

                movement_cost = 1.414

            else:

                movement_cost = 1.0

            new_cost = (
                cost_so_far[current]
                +
                movement_cost
            )

            # ------------------------------------------------
            # BETTER PATH FOUND
            
            if (
                next_point not in cost_so_far
                or
                new_cost < cost_so_far[next_point]
            ):

                cost_so_far[next_point] = new_cost

                priority = (
                    new_cost
                    +
                    distance(
                        next_point,
                        safe_goal
                    )
                )

                heapq.heappush(
                    open_list,
                    (
                        priority,
                        new_cost,
                        next_point
                    )
                )

                came_from[next_point] = current

    # --------------------------------------------------------
    # NO PATH
    print(
        "No path between:",
        safe_start,
        "and",
        safe_goal
    )

    return []

# ============================================================
# STEP 10: SIMPLE ROUTE

start_point = (
    int(start_point[0]),
    int(start_point[1])
)

final_destination = (
    int(end_point[0]),
    int(end_point[1])
)

# ------------------------------------------------------------
# CASUALTY POINTS

casualty_points = [

    casualty["coordinate"]

    for casualty in casualties
]

# ------------------------------------------------------------
# SIMPLE ORDER
# ------------------------------------------------------------

best_order = list(
    range(len(casualty_points))
)

# ------------------------------------------------------------
# CREATE COMPLETE ROUTE

best_route = []

current_point = start_point

valid_route = True

# ============================================================
# VISIT CASUALTIES

for casualty_index in best_order:

    next_point = casualty_points[casualty_index]

    print()
    print(
        "Finding path:",
        current_point,
        "->",
        next_point
    )

    segment = find_safe_path(
        safe_traversability_mask,
        current_point,
        next_point
    )

    # No path found
    if len(segment) == 0:

        print(
            "No safe path to Casualty",
            casualty_index + 1
        )

        valid_route = False

        break
    
    if len(best_route) > 0:

        best_route.extend(
            segment[1:]
        )

    else:

        best_route.extend(
            segment
        )


    current_point = next_point


# ============================================================
# LAST CASUALTY -> FINAL DESTINATION

if valid_route:

    print()
    print(
        "Finding path:",
        current_point,
        "->",
        final_destination
    )

    segment = find_safe_path(
        safe_traversability_mask,
        current_point,
        final_destination
    )

    if len(segment) == 0:

        print(
            "No safe path to final destination."
        )

        valid_route = False

    else:

        if len(best_route) > 0:

            best_route.extend(
                segment[1:]
            )

        else:

            best_route.extend(
                segment
            )

# ============================================================
# DISPLAY ROUTE
print()
print("Simple Rover Route")
print("==================")

if not valid_route or len(best_route) == 0:

    print("NO SAFE ROUTE FOUND.")

else:

    print("SAFE ROUTE FOUND!")

    print()

    print("Casualty Visit Order:")

    for number, index in enumerate(
        best_order,
        start=1
    ):

        print(
            "Stop",
            number,
            "-> Casualty",
            index + 1,
            casualty_points[index]
        )

    print()

    print(
        "Start Point:",
        start_point
    )

    print(
        "Final Destination:",
        final_destination
    )

    print(
        "Total Path Points:",
        len(best_route)
    )

    # ========================================================
    # CREATE ROUTE IMAGE

    route_image = image.copy()

    # ========================================================
    # DRAW PATH
    
    for i in range(
        1,
        len(best_route)
    ):

        point1 = best_route[i - 1]

        point2 = best_route[i]

        cv2.line(
            route_image,
            point1,
            point2,
            (0, 0, 0),
            3
        )

    # ========================================================
    # START MARKER

    cv2.circle(
        route_image,
        start_point,
        10,
        (255, 0, 0),
        -1
    )

    cv2.putText(
        route_image,
        "START",
        (
            start_point[0] + 10,
            start_point[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # ========================================================
    # CASUALTY MARKERS
    
    for visit_number, index in enumerate(
        best_order,
        start=1
    ):

        point = casualty_points[index]

        cv2.circle(
            route_image,
            point,
            10,
            (0, 0, 255),
            2
        )

        cv2.putText(
            route_image,
            "C" + str(visit_number),
            (
                point[0] + 10,
                point[1] - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # ========================================================
    # FINAL DESTINATION
    
    safe_final_destination = best_route[-1]

    cv2.circle(
        route_image,
        safe_final_destination,
        10,
        (255, 0, 0),
        -1
    )

    cv2.putText(
        route_image,
        "DESTINATION",
        (
            safe_final_destination[0] + 10,
            safe_final_destination[1]
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    # ========================================================
    # SAVE IMAGE
    
    route_output_path = (
    "output/optimized_rover_route_" +
    os.path.splitext(os.path.basename(image_path))[0] +
    ".png"

    )

    cv2.imwrite(
        route_output_path,
        route_image
    )

    print()

    print(
        "Rover route image saved."
    )

    print(
        "Saved at:",
        route_output_path
    )

    # ========================================================
    # DISPLAY ROUTE
    
    cv2.imshow(
        "Rover Route",
        route_image
    )

# ============================================================
# SHOW OTHER RESULTS

cv2.imshow(
    "Original Image",
    image
)

cv2.imshow(
    "Traversability Mask",
    traversability_mask
)

cv2.imshow(
    "Casualty Detection",
    casualty_detection_image
)

print()
print(
    "Press any key to close windows."
)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# STEP 12: TIME CALCULATION

def get_terrain_speed(x, y):

    h = int(hsv[y, x, 0])
    s = int(hsv[y, x, 1])
    v = int(hsv[y, x, 2])

    # Green terrain

    if 40 <= h <= 80 and s >= 80:

        # Light green
        if v >= 180:

            return 20.0

        # Moderately dark green
        elif v >= 100:

            return 15.0

        # Darkest green
        else:

            return 10.0


    speeds = []

    for dy in range(-2, 3):

        for dx in range(-2, 3):

            nx = x + dx
            ny = y + dy

            if (
                0 <= nx < hsv.shape[1]
                and
                0 <= ny < hsv.shape[0]
            ):

                nh = int(hsv[ny, nx, 0])
                ns = int(hsv[ny, nx, 1])
                nv = int(hsv[ny, nx, 2])

                if 40 <= nh <= 80 and ns >= 80:

                    if nv >= 180:
                        speeds.append(20.0)

                    elif nv >= 100:
                        speeds.append(15.0)

                    else:
                        speeds.append(10.0)

    if speeds:

        return max(
            set(speeds),
            key=speeds.count
        )

    return 20.0

# ------------------------------------------------------------
# CALCULATE TIME

if best_route is not None:

    light_distance = 0.0
    medium_distance = 0.0
    dark_distance = 0.0

    for i in range(
        1,
        len(best_route)
    ):

        x1, y1 = best_route[i - 1]
        x2, y2 = best_route[i]

        distance = math.sqrt(

            (x2 - x1) ** 2
            +
            (y2 - y1) ** 2
        )

        mid_x = int(
            (x1 + x2) / 2
        )

        mid_y = int(
            (y1 + y2) / 2
        )

        speed = get_terrain_speed(
            mid_x,
            mid_y
        )

        if speed == 20.0:

            light_distance += distance

        elif speed == 15.0:

            medium_distance += distance

        else:

            dark_distance += distance

    # --------------------------------------------------------
    # TIME FOR EACH TERRAIN
    light_time = (
        light_distance / 20.0
    )

    medium_time = (
        medium_distance / 15.0
    )

    dark_time = (
        dark_distance / 10.0
    )

    total_time = (
        light_time
        +
        medium_time
        +
        dark_time
    )

    # --------------------------------------------------------
    # PRINT TIME INFORMATION
    
    print()

    print("Rover Time Calculation")
    print("======================")

    print(
        "Light Green Distance:",
        round(light_distance, 2)
    )

    print(
        "Light Green Time:",
        round(light_time, 4),
        "seconds"
    )

    print(
        "Moderately Dark Green Distance:",
        round(medium_distance, 2)
    )

    print(
        "Moderately Dark Green Time:",
        round(medium_time, 4),
        "seconds"
    )

    print(
        "Darkest Green Distance:",
        round(dark_distance, 2)
    )

    print(
        "Darkest Green Time:",
        round(dark_time, 4),
        "seconds"
    )

    print()

    print(
        "TOTAL TRAVEL TIME:",
        round(total_time, 4),
        "seconds"
    )