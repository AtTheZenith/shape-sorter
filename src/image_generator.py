import cv2
import numpy as np
import random
import os
from numba import njit


@njit
def random_point(x, y, margin=30):
    return [x + random.randint(-margin, margin), y + random.randint(-margin, margin)]


def rotate_image(image, angle, background_color):
    height, width, _ = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )


def skew_image(image, background_color):
    height, width, _ = image.shape
    src_pts = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    dst_pts = np.float32([
        random_point(0, 0),
        random_point(width, 0),
        random_point(0, height),
        random_point(width, height),
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )


def precomputed_sin_lut(size, strength):
    return (strength * np.sin(np.linspace(0, 2 * np.pi, size, endpoint=False))).astype(
        np.float32
    )


def precomputed_cos_lut(size, strength):
    return (strength * np.cos(np.linspace(0, 2 * np.pi, size, endpoint=False))).astype(
        np.float32
    )


def warp_image(image, background_color, warp_strength=25, axis="x", lut=None):
    height, width, _ = image.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    if lut is None:
        lut = precomputed_sin_lut(height if axis == "x" else width, warp_strength)
    if axis == "x":
        offset = lut[(y[:, 0].astype(int)) % len(lut)].reshape(-1, 1)
        x += offset
    else:
        offset = lut[(x[0, :].astype(int)) % len(lut)].reshape(1, -1)
        y += offset
    np.clip(x, 0, width - 1, out=x)
    np.clip(y, 0, height - 1, out=y)
    return cv2.remap(
        image,
        x,
        y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )


def random_color():
    return [random.randint(0, 255) for _ in range(3)]


def generate_contrasting_color(shape_color):
    while True:
        bg_color = random_color()
        if all(abs(shape_color[i] - bg_color[i]) >= 10 for i in range(3)):
            return bg_color


def generate_shape_image(shape, size=(400, 400)):
    bg_color = random_color()
    canvas = np.full((size[0], size[1], 3), bg_color, dtype=np.uint8)
    shape_color = generate_contrasting_color(bg_color)

    max_x = size[1] - 1
    max_y = size[0] - 1

    scale = random.uniform(0.45, 1.05)
    angle = random.randint(0, 360)

    buffer = 40
    max_x -= buffer
    max_y -= buffer

    if shape == "circle":
        radius = int((min(size) // 4) * scale)
        offset_x = random.randint(radius + buffer, max_x - radius)
        offset_y = random.randint(radius + buffer, max_y - radius)

        circle_center = (offset_x, offset_y)
        cv2.circle(canvas, circle_center, radius, shape_color, 4)
    elif shape == "square":
        side = int((min(size) // 2) * scale)
        diagonal = int(np.sqrt(2) * side)

        offset_x = random.randint(diagonal // 2 + buffer, max_x - diagonal // 2)
        offset_y = random.randint(diagonal // 2 + buffer, max_y - diagonal // 2)

        top_left = (offset_x - side // 2, offset_y - side // 2)
        bottom_right = (offset_x + side // 2, offset_y + side // 2)
        cv2.rectangle(canvas, top_left, bottom_right, shape_color, 2)
    elif shape == "triangle":
        side = int((min(size) // 2) * scale)
        height = int(np.sqrt(3) / 2 * side)
        base = side

        offset_x = random.randint(base // 2 + buffer, max_x - base // 2)
        offset_y = random.randint(height // 2 + buffer, max_y - height // 2)

        points = np.array(
            [
                [offset_x, offset_y - height // 2],
                [offset_x - base // 2, offset_y + height // 2],
                [offset_x + base // 2, offset_y + height // 2],
            ],
            np.int32,
        )
        cv2.polylines(canvas, [points], isClosed=True, color=shape_color, thickness=4)
    elif shape == "pentagon":
        num_sides = 5
        side = int((min(size) // 3) * scale)
        r = side

        offset_x = random.randint(r + buffer, max_x - r)
        offset_y = random.randint(r + buffer, max_y - r)

        sin_lut = precomputed_sin_lut(num_sides, r)
        cos_lut = precomputed_cos_lut(num_sides, r)

        points = np.array([
            (int(offset_x + cos_lut[i]), int(offset_y + sin_lut[i]))
            for i in range(num_sides)
        ], np.int32)

        cv2.polylines(canvas, [points], isClosed=True, color=shape_color, thickness=4)
    elif shape == "hexagon":
        num_sides = 6
        side = int((min(size) // 3) * scale)
        r = side

        offset_x = random.randint(r + buffer, max_x - r)
        offset_y = random.randint(r + buffer, max_y - r)

        sin_lut = precomputed_sin_lut(num_sides, r)
        cos_lut = precomputed_cos_lut(num_sides, r)

        points = np.array([
            (int(offset_x + cos_lut[i]), int(offset_y + sin_lut[i]))
            for i in range(num_sides)
        ], np.int32)

        cv2.polylines(canvas, [points], isClosed=True, color=shape_color, thickness=4)
    elif shape == "star-1":
        outer_radius = int((min(size) // 3) * scale)
        inner_radius = int(outer_radius * 0.4)

        offset_x = random.randint(outer_radius + buffer, max_x - outer_radius)
        offset_y = random.randint(outer_radius + buffer, max_y - outer_radius)

        sin_outer = precomputed_sin_lut(10, outer_radius)
        cos_outer = precomputed_cos_lut(10, outer_radius)
        sin_inner = precomputed_sin_lut(10, inner_radius)
        cos_inner = precomputed_cos_lut(10, inner_radius)

        points = []
        for i in range(10):
            if i % 2 == 0:
                x = int(offset_x + cos_outer[i])
                y = int(offset_y + sin_outer[i])
            else:
                x = int(offset_x + cos_inner[i])
                y = int(offset_y + sin_inner[i])

            points.append((x, y))

        points = np.array(points, np.int32)
        cv2.polylines(canvas, [points], isClosed=True, color=shape_color, thickness=4)

    elif shape == "star-2":
        outer_radius = int((min(size) // 3) * scale)
        inner_radius = int(outer_radius * 0.5)

        offset_x = random.randint(outer_radius + buffer, max_x - outer_radius)
        offset_y = random.randint(outer_radius + buffer, max_y - outer_radius)

        sin_outer = precomputed_sin_lut(5, outer_radius)
        cos_outer = precomputed_cos_lut(5, outer_radius)
        sin_inner = precomputed_sin_lut(5, inner_radius)
        cos_inner = precomputed_cos_lut(5, inner_radius)

        points = []
        for i in range(5):
            points.append((
                int(offset_x + cos_outer[i]),
                int(offset_y + sin_outer[i])
            ))
            points.append((
                int(offset_x + cos_inner[i] * np.cos(np.pi / 5)),
                int(offset_y + sin_inner[i] * np.cos(np.pi / 5))
            ))

        points = np.array(points, np.int32)

        for i in range(0, 10, 2):
            cv2.line(canvas, tuple(points[i]), tuple(points[(i + 4) % 10]), color=shape_color, thickness=2)


    rotated_canvas = rotate_image(canvas, angle, bg_color)
    skewed_canvas = skew_image(rotated_canvas, bg_color)
    warped_canvas = warp_image(
        skewed_canvas, bg_color, warp_strength=10, axis=random.choice(["x", "y"])
    )
    return warped_canvas


def save_images(num_images=10, output_folder="shapes"):
    os.makedirs(output_folder, exist_ok=True)
    shapes = ["circle", "square", "triangle", "pentagon", "hexagon", "star-1", "star-2"]
    shape_counts = {
        "circle": 0,
        "square": 0,
        "triangle": 0,
        "pentagon": 0,
        "hexagon": 0,
        "star-1": 0,
        "star-2": 0,
    }

    for shape in shapes:
        os.makedirs(os.path.join(output_folder, shape), exist_ok=True)

    for _ in range(num_images):
        shape = random.choice(shapes)
        shape_counts[shape] += 1
        img = generate_shape_image(shape)
        filename = os.path.join(output_folder, shape, f"{shape_counts[shape]}.png")
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")


if __name__ == "__main__":
    save_images(30000, "training")
    save_images(1000, "testing")
