
from pathlib import Path
import cv2
import numpy as np

def compute_2d_fft(work_dir, puzzle_name, image, puzzle_piece):
    output_dir = Path(work_dir, puzzle_name)
    image_output_dir = Path(output_dir, "fourier")
    Path(image_output_dir).mkdir(parents = True, exist_ok = True)

    filtered = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Laplacian(filtered, cv2.CV_64F)
    # Find contours
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #draw contours and use it for the fourier transform
    drawing = np.zeros_like(image)
    #draw contours filled
    cv2.drawContours(drawing, contours[0], -1, 255, thickness=cv2.FILLED)
    # circular low pass filter
    f = np.fft.fft2(drawing)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    # save as image
    #show in window drawing
    cv2.imwrite(Path(image_output_dir, f"{puzzle_piece}.png"), magnitude_spectrum)
    cv2.imwrite(Path(image_output_dir, f"{puzzle_piece}_original.png"), drawing)
    # save the magnitude spectrum to a file
    np.save("magnitude_spectrum.npy", magnitude_spectrum)
    '''with open(Path(image_output_dir, f"{puzzle_piece}.txt"), "w") as file:
        for row in magnitude_spectrum:
            file.write(", ".join([str(x) for x in row]) + "\n")'''
    