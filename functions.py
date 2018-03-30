import numpy as np

def postfilter(X, change_points):
    if len(change_points) > 0:
        change_points = detect_local_lines(X, change_points, np.pi / 90, 5)
    return change_points

# pretty janky but works for us so I guess its good
def detect_local_lines(Z, change_points, threshold_angle, radius):
    X = Z * 3
    n = len(X)
    char_n = len(X[0])
    new_ch_points = []
    res_angles = []

    half_radius = int(radius / 2)

    for change_point in change_points:
        left = change_point - 10
        right = change_point + 10
        if left < 0:
            left = 0
        if right >= n:
            right = n - 1

        coef_l = np.ones((char_n))
        coef_r = np.ones((char_n))
        angles = []
        ang_points = []
        left_point = change_point
        right_point = change_point

        tmp = np.zeros((char_n))

        for point in range(left, change_point):
            for j in range(char_n):
                if point >= half_radius:
                    l = point - radius
                    if l < 0:
                        l = 0

                    weights = np.zeros((point + 1 - l))
                    weights[-1] = 1
                    weights[:-1] = 0.01

                    coef = np.polynomial.polynomial.polyfit(range(l, point + 1), X[l: point + 1, j], 1, w=weights)
                    coef_l[j] = coef[1]

                    del weights
                if point < n - 3:
                    r = point + 5
                    if r >= n:
                        r = n - 1

                    weights = np.zeros((r + 1 - point))
                    weights[0] = 1
                    weights[1:] = 0.01

                    coef = np.polynomial.polynomial.polyfit(range(point, r + 1), X[point: r + 1, j], 1, w=weights)
                    coef_r[j] = coef[1]
                    del weights
            coef_l = np.arctan(coef_l)
            coef_r = np.arctan(coef_r)
            for j in range(char_n):
                angle = abs(coef_l[j] - coef_r[j])
                if angle > np.pi / 2:
                    angle = np.pi - angle
                tmp[j] = angle
            angles.append(np.average(tmp))
            ang_points.append(point)
        zipped = zip(angles, ang_points)
        zipped = sorted(zipped, reverse=True)
        # print(zipped)
        l_angle = 0
        if zipped[0][0] > threshold_angle:
            left_point = zipped[0][1]
            l_angle = zipped[0][0]

        coef_l[:] = 1
        coef_r[:] = 1
        angles = []
        ang_points = []
        tmp = np.zeros((char_n))
        for point in range(change_point, right + 1):
            for j in range(char_n):
                if point >= half_radius:
                    l = point - radius
                    if l < 0:
                        l = 0
                    coef = np.polynomial.polynomial.polyfit(range(l, point + 1), X[l: point + 1, j], 1)
                    coef_l[j] = coef[1]
                if point < n - half_radius - 1:
                    r = point + radius
                    if r >= n:
                        r = n - 1
                    coef = np.polynomial.polynomial.polyfit(range(point, r + 1), X[point: r + 1, j], 1)
                    coef_r[j] = coef[1]
            coef_l = np.arctan(coef_l)
            coef_r = np.arctan(coef_r)

            for j in range(char_n):
                angle = abs(coef_l[j] - coef_r[j])
                if angle > np.pi / 2:
                    angle = np.pi - angle
                tmp[j] = angle
            angles.append(np.average(tmp))
            ang_points.append(point)

        zipped = zip(angles, ang_points)
        zipped = sorted(zipped, reverse=True)
        # print(zipped)
        r_angle = 0
        if zipped[0][0] > threshold_angle:
            right_point = zipped[0][1]
            r_angle = zipped[0][0]

        if left_point == right_point:
            new_ch_points.append(left_point)
            res_angles.append(l_angle)
        else:
            if left_point not in new_ch_points:
                new_ch_points.append(left_point)
                res_angles.append(l_angle)
            if right_point not in new_ch_points:
                new_ch_points.append(right_point)
                res_angles.append(r_angle)
    zipped = zip(new_ch_points, res_angles)
    zipped = sorted(zipped)
    new_ch_points = [x[0] for x in zipped]
    res_angles = [x[1] for x in zipped]
    # new_ch_points = remove_close(new_ch_points, res_angles)
    return new_ch_points
