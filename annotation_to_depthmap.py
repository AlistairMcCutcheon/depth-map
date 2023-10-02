import json
import numpy as np
import math
import cv2

EARTH_RADIUS = 6371000
NUM_REFERENCE_LINES = 9
MAVIC_CAMERA_DIAG_FOV_DEG = 77


# Draw horizontal line where horizon should be.
# horizon_pixel is the number of pixels from the top where the horizon should be
def draw_horizon_line(img: np.ndarray, horizon_pixel: int) -> np.ndarray:
    height, width = img.shape[:2]
    return cv2.line(
        img,
        (0, int((1 - horizon_pixel) * height)),
        (width, int((1 - horizon_pixel) * height)),
        color=(0, 0, 255),
        thickness=20,
    )


# Draw reference lines across the image to aid visualisation
def draw_reference_lines(img: np.ndarray) -> np.ndarray:
    height, width = img.shape[:2]
    for i in range(1, (NUM_REFERENCE_LINES + 1)):
        img = cv2.line(
            img,
            (0, int(i / (NUM_REFERENCE_LINES + 1) * height)),
            (width, int(i / (NUM_REFERENCE_LINES + 1) * height)),
            color=(0, 0, 0),
            thickness=10,
        )
    return img


def calc_fovs(img_width: int, img_height: int) -> tuple[float, float, float]:
    fov_diag = MAVIC_CAMERA_DIAG_FOV_DEG * math.pi / 180
    focal_length_pixels = (math.sqrt(img_width**2 + img_height**2) / 2) / math.tan(
        fov_diag / 2
    )
    fov_v = 2 * math.atan(2160 / 2 / focal_length_pixels)
    fov_h = 2 * math.atan(3840 / 2 / focal_length_pixels)
    return fov_diag, fov_h, fov_v


def calc_upper_lower_angles(pitch, fov_v):
    upper_angle = pitch - fov_v / 2
    lower_angle = pitch + fov_v / 2
    return upper_angle, lower_angle


# returns the number of pixels from the top where the horizon should be
def calc_horizon_pixel(pitch, fov_v, altitude_m) -> float:
    upper_angle, lower_angle = calc_upper_lower_angles(pitch, fov_v)

    angle_to_horizon = math.pi / 2 - math.asin(
        EARTH_RADIUS / (EARTH_RADIUS + altitude_m)
    )
    return (angle_to_horizon - lower_angle) / (upper_angle - lower_angle)


# create new image with line where horizon should be
def create_line_img(img: np.ndarray, image_annotation: dict):
    altitude_m = image_annotation["meta"]["height_above_takeoff(meter)"]
    pitch_deg = image_annotation["meta"]["gimbal_pitch(degrees)"]
    pitch_rad = (pitch_deg / 180) * math.pi
    _, _, fov_v = calc_fovs(image_annotation["width"], image_annotation["height"])

    horizon_pixel = calc_horizon_pixel(pitch_rad, fov_v, altitude_m)

    line_img = draw_horizon_line(img, horizon_pixel)
    line_img = draw_reference_lines(line_img)
    return line_img


# calc distance to pixel, where pixel is the y value of the pixel in the image
def calc_distance_to_pixel(pitch, fov_v, altitude_m, pixel, img_height):
    upper_angle, lower_angle = calc_upper_lower_angles(pitch, fov_v)

    angle = lower_angle + ((upper_angle - lower_angle) / img_height) * pixel
    distance = altitude_m * math.tan(math.pi / 2 - angle)
    return distance


# create greyscale image where the pixel values are the normalised distances to each pixel in the original img
def create_dist_image(img: np.ndarray, image_annotation: dict):
    height, width = img.shape[:2]
    distances = np.zeros((height, 1))

    altitude_m = image_annotation["meta"]["height_above_takeoff(meter)"]
    pitch_deg = image_annotation["meta"]["gimbal_pitch(degrees)"]
    pitch_rad = (pitch_deg / 180) * math.pi
    _, _, fov_v = calc_fovs(image_annotation["width"], image_annotation["height"])
    for i in range(height):
        distances[i] = calc_distance_to_pixel(pitch_rad, fov_v, altitude_m, i, height)
    dist_img = np.tile(np.flip(distances, axis=0), (1, width))
    dist_img_normalised = dist_img / np.max(dist_img) * 255
    return dist_img_normalised


def main():
    with open("data/SeaDronesSee/Annotations/instances_val.json") as file:
        data = json.load(file)

    for annotation in data["images"]:
        if not annotation["source"]["drone"] == "mavic":
            continue

        image_id = annotation["id"]
        img = cv2.imread(f"data/SeaDronesSee/Images/val/{image_id}.jpg").astype(
            np.float32
        )
        line_img = create_line_img(img, annotation)
        cv2.imwrite(f"line_imgs/{image_id}.png", line_img)

        dist_img = create_dist_image(img, annotation)
        cv2.imwrite(f"dist_imgs/{image_id}.png", dist_img)


if __name__ == "__main__":
    main()
