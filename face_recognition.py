# Edge Impulse - OpenMV FOMO Object Detection Example
#
# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE

import sensor, image, time, ml, math, uos, gc

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(time=2000)          # Let the camera adjust.

net = None
labels = None
min_confidence = 0.8

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained_equal_imgs.tflite", load_to_fb=uos.stat('trained_equal_imgs.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained_equal_imgs.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

threshold_list = [(math.ceil(min_confidence * 255), 255)]

# --- CSV Logging Setup ---
csv_filename = "face_detections.csv"
csv_file = None


try:
    # Check if file exists to write header
    file_exists = False
    try:
        uos.stat(csv_filename)
        file_exists = True
    except OSError:
        file_exists = False

    # Open file in append mode
    # This will save to internal flash if no SD card is present and script is run from flash
    csv_file = open(csv_filename, "a")

    # Write header if file is new
    if not file_exists:
        csv_file.write("timestamp,label,score\n")
        csv_file.flush() # Ensure header is written immediately
    print(f"CSV file {csv_filename} opened for appending on internal flash (or SD card if present).")
    print("Warning: Internal flash storage is limited. Consider using an SD card for extended logging.")

except Exception as e:
    print(f"Error opening or writing to CSV file {csv_filename}: {e}")
    # Continue without logging if file access fails
    csv_file = None # Ensure csv_file is None if opening failed

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

# Track when we first saw each person label
detection_start = {}   # label -> timestamp_ms
confirmed = set()      # labels confirmed (≥1 s)
confirm_time_ms = 1000 # how long to see them before confirming ## 1 second

detections_for_logging = []

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    now = time.ticks_ms()

    seen_this_frame = set()

    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0: continue  # background class
        if len(detection_list) == 0: continue  # no detections for this class?

        #print("********** %s **********" % labels[i])

        label = labels[i]
        seen_this_frame.add(label)


        for x, y, w, h, score in detection_list:

            # draw names on live video
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            img.draw_string(0,0,labels[i],color=colors[i])

            # start timers
            for label in seen_this_frame:
                if label not in detection_start:
                    detection_start[label] = now


            # check if 1 second has elapsed
            for label in list(detection_start):
                if label in seen_this_frame:
                    elapsed = time.ticks_diff(now, detection_start[label])
                    if elapsed >= confirm_time_ms and label not in confirmed:
                        confirmed.add(label)
                        print("✅ Confirmed:", label)

                        # write to dict
                        detections_for_logging.append({
                                     "Time": elapsed,
                                     "Name": label,
                                     "Confidence": score,
                                 })

                        # write each detection to the csv
                        for det in detections_for_logging:

                            csv_line = f"{det['Time']},{det['Name']},{det['Confidence']:.2f}\n"
                            csv_file.write(csv_line)

                            print(f"Logged detection(s) to {csv_filename}")
                else:
                    # lost track: reset timer
                    del detection_start[label]
    csv_file.flush()


    if confirmed:
        # list of names + "attendance" to draw all at once
        names = ["Attendance:"] + sorted(confirmed)
        # positioning
        max_chars = max(len(n) for n in names)
        x_pos = img.width() - (max_chars * 8)

        # draw each line, 10 px apart
        for i, n in enumerate(names):
            y = i * 10
            img.draw_string(x_pos, y, n, color=(255,255,255))
