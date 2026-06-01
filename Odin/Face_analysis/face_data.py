def extract_face_data(landmarks):
    # Estimated trichion (hairline). MediaPipe does not model the scalp:
    # landmark 10 is the topmost mesh vertex and sits on the upper forehead,
    # biased below the true hairline. We extend the glabella -> forehead_top
    # vector upward so the estimate scales with face size and follows head
    # tilt (a fixed pixel offset does neither). TRICHION_K is tuned visually
    # against the debug overlays.
    TRICHION_K = 0.45
    forehead_top = landmarks[10]
    glabella = (landmarks[8] + landmarks[9]) / 2
    trichion = forehead_top + TRICHION_K * (forehead_top - glabella)

    face_data = {
            # Left Eye Contour
            "left_upper_eyelid_center": landmarks[159],
            "left_lower_eyelid_center": landmarks[145],
            "left_eye_outer_corner": landmarks[33],
            "left_eye_inner_corner": landmarks[133],

            # Right Eye Contour
            "right_upper_eyelid_center": landmarks[386],
            "right_lower_eyelid_center": landmarks[374],
            "right_eye_outer_corner": landmarks[263],
            "right_eye_inner_corner": landmarks[362],

            #Left Pupil
            "left_pupil_center": landmarks[468],

            #Right Pupil 
            "right_pupil_center": landmarks[473],

            # Left Eyebrow
            "left_eyebrow_upper_outer_point": landmarks[70],
            "left_eyebrow_upper_inner_point": landmarks[107],
            "left_eyebrow_lower_outer_point": landmarks[46],
            "left_eyebrow_lower_inner_point": landmarks[55],
            "left_eyebrow_peak_from_eye": landmarks[52],
            "left_eyebrow_peak_from_forehead": landmarks[105],

            # Right Eyebrow
            "right_eyebrow_upper_outer_point": landmarks[300],
            "right_eyebrow_upper_inner_point": landmarks[336],
            "right_eyebrow_lower_outer_point": landmarks[276],
            "right_eyebrow_lower_inner_point": landmarks[285],
            "right_eyebrow_peak_from_eye": landmarks[282],
            "right_eyebrow_peak_from_forehead": landmarks[334],

            "eyebrows_bottom": (
                landmarks[55] + 
                landmarks[285] + 
                landmarks[46] + 
                landmarks[276]
            ) / 4,

            # Nose
            "base_of_nose": landmarks[2],
            "top_of_nose_bridge": landmarks[168],
            "nose_tip": landmarks[4],
            # Alare wing points: 48/278 (alar base). Switched from 102/331,
            # which sat inconsistently medial/lateral and gave mixed nose widths.
            "left_alare_tip": landmarks[48],
            "right_alare_tip": landmarks[278],

            # Lips
            "lip_left_outer": landmarks[61],
            "lip_right_outer": landmarks[291],
            "upper_lip_top_center": landmarks[0],
            "upper_lip_bottom_center": landmarks[13],
            "lower_lip_top_center": landmarks[14],
            "lower_lip_bottom_center": landmarks[17],

            # Jawline and Oval
            "chin": landmarks[152],
            "left_zygomatic": landmarks[234],
            "right_zygomatic": landmarks[454],
            "left_jaw_angle1": landmarks[132],
            "left_jaw_angle2": landmarks[58],
            "right_jaw_angle1": landmarks[361],
            "right_jaw_angle2": landmarks[288],
            "left_jaw_bottom": landmarks[172],
            "right_jaw_bottom": landmarks[397],

            # Forehead
            # top_center_forehead is the estimated trichion, not raw landmark 10.
            "top_center_forehead": trichion,
            "glabella": glabella,

            # Cheeks
            "left_cheek_apex": landmarks[50],
            "left_cheek_hollow": landmarks[205],
            "right_cheek_apex": landmarks[280],
            "right_cheek_hollow": landmarks[425]
        }
    
    return face_data
    
