# Digitizes Fig. 1 of Giudice et al. (2019), Ann. Biomed. Eng. 47:1855-1872,
# the six-DOF head kinematics applied to SIMon, from the JPEG embedded in the
# PDF (pdfimages page 5, image 0, 2032 x 805 px at 300 ppi, saved as PPM).
#
# Each panel is calibrated by its spines: the left spine spans the y range
# of the axis (-6000 to 8000 mm/s, -60 to 10 rad/s), the bottom spine the
# time 0 to 0.06 s. A curve is the pixels of its color (black, red, blue)
# inside the axes, outside the legend; in each column its center is the
# mean row of the run of that color nearest the previous column. Columns
# where a curve is hidden under another one are filled linearly.
#
# Output, in the folder it runs in:
#    fig1_curves.csv     t [s], vx, vy, vz [mm/s], wx, wy, wz [rad/s], every
#                        1e-4 s, smoothed over 5 px
#    fig1_curves.k       the same as *DEFINE_CURVE 3-8, for
#                        LoadData::read_motion of brain_simon
#    fig1_overlay.png    the digitized curves drawn over the figure
#
# Run, in the folder of the image:
#    pdfimages -f 5 -l 5 ABE-2018-GIUDICE.pdf page5
#    python3 -I digitize_fig1.py page5-000.ppm
import sys
import zlib
import struct

import numpy as np


def read_ppm(path):
    with open(path, 'rb') as file:
        data = file.read()
    fields = []
    pos = 0
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(data[start:pos])
    pos += 1
    width, height = int(fields[1]), int(fields[2])
    return np.frombuffer(data[pos:pos + width * height * 3], np.uint8).reshape(height, width, 3)


def write_png(path, image):
    height, width, _ = image.shape
    raw = b''.join(b'\x00' + image[row].tobytes() for row in range(height))

    def chunk(kind, body):
        return (struct.pack('>I', len(body)) + kind + body
                + struct.pack('>I', zlib.crc32(kind + body) & 0xffffffff))

    with open(path, 'wb') as file:
        file.write(b'\x89PNG\r\n\x1a\n'
                   + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                   + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))


def longest_run(line):
    """Start and end of the longest run of True in line."""
    best, start, best_span = 0, None, (0, -1)
    for index, value in enumerate(np.append(line, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > best:
                best, best_span = index - start, (start, index - 1)
            start = None
    return best_span


def find_spines(dark, col_range):
    """The left spine column and its rows, the bottom spine row and its
    columns, as the longest straight dark lines in the columns col_range;
    the spine is the innermost (rightmost, lowest) column or row of it."""
    lo, hi = col_range
    col_spans = [longest_run(dark[:, col]) for col in range(lo, hi)]
    col_lengths = np.array([span[1] - span[0] for span in col_spans])
    left_cols = np.where(col_lengths >= col_lengths.max() - 3)[0]
    left = lo + int(left_cols.max())
    # The curves cross the spine at t = 0, which cuts its run; its top is
    # the first dark pixel of the column, its bottom that of the run.
    rows = (int(np.where(dark[:, left])[0].min()), col_spans[left - lo][1])
    row_spans = [longest_run(dark[row, lo:hi]) for row in range(dark.shape[0])]
    row_lengths = np.array([span[1] - span[0] for span in row_spans])
    bottom_rows = np.where(row_lengths >= row_lengths.max() - 3)[0]
    bottom = int(bottom_rows.min())
    cols = (lo + row_spans[bottom][0], lo + row_spans[bottom][1])
    return left, rows, bottom, cols


def color_masks(image):
    rr, gg, bb = (image[:, :, kk].astype(int) for kk in range(3))
    return {
        'black': (rr < 90) & (gg < 90) & (bb < 90),
        'red': (rr > 170) & (gg < 110) & (bb < 110),
        'blue': (bb > 170) & (rr < 110) & (gg < 140),
    }


def trace(mask, cols, row_lo, row_hi, start_row):
    """Center row of the curve in each column, NaN where it is not seen."""
    centers = np.full(len(cols), np.nan)
    previous = start_row
    for index, col in enumerate(cols):
        rows = np.where(mask[row_lo:row_hi, col])[0] + row_lo
        if rows.size == 0:
            continue
        breaks = np.where(np.diff(rows) > 2)[0]
        runs = np.split(rows, breaks + 1)
        best = min(runs, key=lambda run: abs(run.mean() - previous))
        if abs(best.mean() - previous) > 40 and not np.isnan(centers).all():
            continue
        centers[index] = best.mean()
        previous = centers[index]
    seen = ~np.isnan(centers)
    centers[~seen] = np.interp(np.where(~seen)[0], np.where(seen)[0], centers[seen])
    return centers


def main():
    image = read_ppm(sys.argv[1])
    height, width, _ = image.shape
    masks = color_masks(image)
    # The spines are thin gray lines, about 120-150 at their center.
    channels = image.astype(int)
    dark = (channels.max(axis=2) - channels.min(axis=2) < 30) & (channels.mean(axis=2) < 190)

    # The two panels, left and right halves; their y ranges and legends,
    # (col_lo, col_hi, row_lo, row_hi) of the legend box with a margin.
    panels = [
        {'cols': (0, width // 2), 'y_range': (-6000.0, 8000.0),
         'names': ['vx', 'vy', 'vz'], 'legend': (150, 300, 510, 690)},
        {'cols': (width // 2, width), 'y_range': (-60.0, 10.0),
         'names': ['wx', 'wy', 'wz'], 'legend': (1170, 1320, 510, 690)},
    ]
    t_out = np.round(np.arange(0.0, 0.06 + 1.0e-9, 1.0e-4), 6)
    curves = {}
    overlay = image.copy()
    for panel in panels:
        left, (top_row, bottom_row), bottom, (left_col, right_col) = \
            find_spines(dark, panel['cols'])
        y_min, y_max = panel['y_range']
        print(f"panel {panel['names']}: left spine col {left}, rows {top_row}-{bottom_row}; "
              f"bottom spine row {bottom}, cols {left_col}-{right_col}")
        col_to_t = lambda col: (col - left) / (right_col - left) * 0.06
        row_to_y = lambda row: y_min + (bottom - row) / (bottom - top_row) * (y_max - y_min)
        y_to_row = lambda value: bottom - (value - y_min) / (y_max - y_min) * (bottom - top_row)

        cols = np.arange(left + 4, right_col - 1)
        legend = panel['legend']
        for name, color in zip(panel['names'], ['black', 'red', 'blue']):
            mask = masks[color].copy()
            mask[:, :left + 4] = False
            mask[bottom - 3:, :] = False
            mask[legend[2]:legend[3], legend[0]:legend[1]] = False
            centers = trace(mask, cols, top_row, bottom - 3, y_to_row(0.0))
            # Smooth over 5 px, then sample every 1e-4 s.
            kernel = np.ones(5) / 5.0
            padded = np.concatenate([np.full(2, centers[0]), centers, np.full(2, centers[-1])])
            smooth = np.convolve(padded, kernel, mode='valid')
            values = row_to_y(smooth)
            times = col_to_t(cols)
            curves[name] = np.interp(t_out, times, values)
            for col, row in zip(cols, smooth):
                rr = int(round(row))
                overlay[rr - 1:rr + 2, col] = (0, 200, 0)

    columns = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
    with open('fig1_curves.csv', 'w') as file:
        file.write('t_s,vx_mm_s,vy_mm_s,vz_mm_s,wx_rad_s,wy_rad_s,wz_rad_s\n')
        for index, time in enumerate(t_out):
            file.write(f'{time:.4f},' + ','.join(f'{curves[name][index]:.6g}' for name in columns)
                       + '\n')

    with open('fig1_curves.k', 'w') as file:
        file.write('*KEYWORD\n$ Fig. 1 of Giudice et al. (2019), digitized by digitize_fig1.py:\n'
                   '$ curves 3-5 velocity of the head center in mm/s, 6-8 angular velocity\n'
                   '$ in rad/s, about the global axes as in Simon.k.\n')
        for lcid, name in zip(range(3, 9), columns):
            file.write(f'*DEFINE_CURVE\n    {lcid},0,1,1.0\n')
            for time, value in zip(t_out, curves[name]):
                file.write(f'{time:20.6e}{value:20.6e}\n')
        file.write('*END\n')

    write_png('fig1_overlay.png', overlay)
    for name in columns:
        peak = int(np.argmax(np.abs(curves[name])))
        print(f'{name}: at 0 {curves[name][0]:9.3f}, peak {curves[name][peak]:9.3f} '
              f'at {t_out[peak]:.4f} s, at 0.06 {curves[name][-1]:9.3f}')


main()
