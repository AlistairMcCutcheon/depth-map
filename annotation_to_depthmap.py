import json
import numpy as np
import math
import cv2
import tifffile

EARTH_RADIUS = 6371000
NUM_REFERENCE_LINES = 9


# Draw horizontal line where horizon should be.
# horizon_pixel is the number of pixels from the top where the horizon should be
def draw_horizon_line(img: np.ndarray, horizon_pixel: int):
    height, width = img.shape[:2]
    return cv2.line(
        img,
        (0, int((1 - horizon_pixel) * height)),
        (width, int((1 - horizon_pixel) * height)),
        color=(0, 0, 255),
        thickness=20,
    )


# Draw reference lines across the image to aid visualisation
def draw_reference_lines(img):
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


def create_line_img(image_annotation: dict):
    image_id = image_annotation["id"]
    img_height = image_annotation["height"]
    img_width = image_annotation["width"]

    fov_diag = 77 * math.pi / 180
    focal_length_pixels = (math.sqrt(img_width**2 + img_height**2) / 2) / math.tan(
        fov_diag / 2
    )
    fov_v = 2 * math.atan(2160 / 2 / focal_length_pixels)
    fov_h = 2 * math.atan(3840 / 2 / focal_length_pixels)

    altitude_m = image_annotation["meta"]["height_above_takeoff(meter)"]

    pitch_deg = image_annotation["meta"]["gimbal_pitch(degrees)"]
    pitch_rad = (pitch_deg / 180) * math.pi

    upper_angle = pitch_rad - fov_v / 2
    lower_angle = pitch_rad + fov_v / 2

    dist_to_horizon = math.sqrt(2 * EARTH_RADIUS * altitude_m + altitude_m**2)

    angle_to_horizon = math.pi / 2 - math.asin(
        EARTH_RADIUS / (EARTH_RADIUS + altitude_m)
    )

    horizon_pixel = (angle_to_horizon - lower_angle) / (upper_angle - lower_angle)

    img = cv2.imread(f"data/SeaDronesSee/Images/train/{image_id}.jpg").astype(
        np.float32
    )

    line_img = draw_horizon_line(img, horizon_pixel)
    line_img = draw_reference_lines(line_img)
    cv2.imwrite(f"line_imgs/{image_id}_{pitch_deg}.png", line_img)

    print()
    print(f"Image Id: {image_id}")
    print(f"Image Width: {img_width}")
    print(f"Image Height: {img_height}")
    print(f"FOV Diagonal: {fov_diag}")
    print(f"FOV Horizontal: {fov_h}")
    print(f"FOV Vertical: {fov_v}")
    print("Altitude:", altitude_m)
    print("Pitch (deg):", pitch_deg)
    print("Pitch (rad):", pitch_rad)
    print("Lower angle (deg):", lower_angle * 180 / math.pi)
    print("Upper angle (deg):", upper_angle * 180 / math.pi)
    print("Lower angle (rad):", lower_angle)
    print("Upper angle (rad):", upper_angle)
    print("Dist to horison:", dist_to_horizon)
    print("Angle to horizon (rad):", angle_to_horizon)
    print("Angle to horizon (deg):", angle_to_horizon * 180 / math.pi)


def main():
    with open("data/SeaDronesSee/Annotations/instances_train.json") as file:
        data = json.load(file)

    for annotation in data["images"]:
        if not annotation["source"]["drone"] == "mavic":
            continue
        create_line_img(annotation)


if __name__ == "__main__":
    main()


# # image_id = 3453
# image_id = 7386
# # image_id = 7416
# # image_id = 8258
# image_id = 8348

# if annotation["id"] == image_id:
#     image_annotation = annotation
#     break

# print(image_annotation)


# EARTH_RADIUS = 6378137


# img = cv2.imread(
#     "/home/alistair/Downloads/Compressed Version/images/train/10736.jpg"
# ).astype(np.float32)


# distances = np.zeros((img_height, 1))
# for i in range(img_height):
#     angle = lower_angle + ((upper_angle - lower_angle) / img_height) * i
#     distance = altitude_m * math.tan(math.pi / 2 - angle)
#     distances[i] = distance

# print("Distances:")
# for i, x in enumerate(distances):
#     if i % 10 == 0:
#         print(x)
# print(distances)

# dist_img = np.tile(np.flip(distances, axis=0), (1, img_width))
# dist_img_normalised = dist_img / np.max(dist_img) * 255
# cv2.imwrite("dist_img.png", dist_img_normalised)

# weighted_img = img * np.expand_dims(dist_img_normalised / 255, 2)
# cv2.imwrite("dist_img_multiplied.png", weighted_img)

# diff_differences = np.zeros((img_height, 1))
# for i in range(img_height):
#     angle = lower_angle + ((upper_angle - lower_angle) / img_height) * i
#     distance = altitude_m * math.tan(math.pi / 2 - angle)
#     angle_next = lower_angle + ((upper_angle - lower_angle) / img_height) * (i + 1)
#     distance_next = altitude_m * math.tan(math.pi / 2 - angle_next)
#     diff_differences[i] = distance / distance_next
# diff_differences_img = np.tile(np.flip(diff_differences, axis=0), (1, img_width))
# diff_differences_img_normalised = (
#     diff_differences_img - np.min(diff_differences_img)
# ) / (np.max(diff_differences_img) - np.min(diff_differences_img))


# cv2.imwrite("diff_dist_img_normalised.png", diff_differences_img_normalised * 255)
# img = cv2.imread(
#     "/home/alistair/Downloads/Compressed Version/images/train/10736.jpg"
# ).astype(np.float32)
# img = img * np.expand_dims(diff_differences_img, 2)
# cv2.imwrite("diff_dist_img_multiplied.png", img)
