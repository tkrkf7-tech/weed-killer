# weed-killer
AI robot on Raspberry Pi performs camera inference, detects weeds, lights Arduino LED, and motion
#========================
# Essential setting import
#=========================
import time
import sys
import os
import json
import subprocess
import cv2
from smbus2 import SMBus
import serial
# =========================
# (A) PCA9685(Motro name) / I2C setting(to control the motor) 
# =========================
PCA_ADDR = 0x5F   	# Present Board(i2cdetect Standard)
I2C_BUS = 1
# Motor Channel
LEFT_IN1  = 8
LEFT_IN2  = 9
RIGHT_IN1 = 11
RIGHT_IN2 = 10
# ==========================================
# (B). Arduino Serial (LED) 
# ==========================================
SER_PORT = "/dev/ttyACM0"  
SER_BAUD = 9600

try:
    arduino = serial.Serial(SER_PORT, SER_BAUD, timeout=1)
    time.sleep(2)
except Exception as e:
    print("[Arduino] connectino failure:", e)
    arduino = None

def notify_arduino(cmd):
    if arduino is None:
        return
    try:
        arduino.write((cmd + "\n").encode("utf-8"))
    except:
        pass
# Actual Arduino on-off command       
def led_on():
    notify_arduino("WEED")

def led_off():
    notify_arduino("CLEAR")

# =========================
# (C) Servo(Adeept RPIservo)
#Because the robot is from Adeept, so we used the Adeept & RPIservo from Adeept
# =========================
sys.path.append('/home/wolf/Adeept_PiCar-Pro/Server')
from RPIservo import ServoCtrl
sc = ServoCtrl()
SERVO_A = 0  #for front wheel alignment
SERVO_B = 1  # pan
SERVO_C = 2  # tilt
SERVO_D = 3  # fixed
CENTER = 90
PAN_DELTA = 55
LEFT_ANGLE  = CENTER - PAN_DELTA   # -45
RIGHT_ANGLE = CENTER + PAN_DELTA   # +45
TILT_ANGLE = 45
D_ANGLE	= 30
SERVO_SETTLE = 2.5

# =========================
# (C) Movement parameter
# =========================
MOTOR_SPEED_PERCENT = 40   
MOVE_TIME  = 0.50       	# Step 4 Advance time / Speed ​​may vary depending on location and battery condition
STOP_TIME  = 0.3       	# Step 1 Pausing time
# =========================
# (D) Inference Calling(Inference venv: wolfverry)
# =========================
INFER_PY = "/home/wolf/wolfverry/bin/python"
#AI library in venv file for inference 
INFER_SCRIPT = "/home/wolf/infer_tflite.py"
#Inference file
# No matter if labels.txt is Korean(딸기/잡초), it process pred index
STRAW_PRED = 0  # 0 = 딸기
WEED_PRED  = 1  # 1 = 잡초
# =========================
# (E) Direct control PWM by smbus2
#  To make rasberry control the speed and stop of motor through PCA9685 accurately
# =========================
LED0_ON_L = 0x06
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
def pca_write8(bus, reg, val):
    bus.write_byte_data(PCA_ADDR, reg, val & 0xFF)
def pca_set_pwm(bus, ch, on, off):
    base = LED0_ON_L + 4 * ch
    pca_write8(bus, base + 0, on & 0xFF)
    pca_write8(bus, base + 1, (on >> 8) & 0xFF)
    pca_write8(bus, base + 2, off & 0xFF)
    pca_write8(bus, base + 3, (off >> 8) & 0xFF)
def pca_set_duty(bus, ch, duty_12bit):
    duty_12bit = clamp(int(duty_12bit), 0, 4095)
    pca_set_pwm(bus, ch, 0, duty_12bit)
def motor_stop(bus):
    for ch in (LEFT_IN1, LEFT_IN2, RIGHT_IN1, RIGHT_IN2):
        pca_set_duty(bus, ch, 0)
def motor_forward(bus, speed_percent=40, duration=1.0):
    duty = int(clamp(speed_percent, 0, 100) * 4095 / 100)
    # Forward movement: IN1=PWM, IN2=0
    pca_set_duty(bus, LEFT_IN1, duty)
    pca_set_duty(bus, LEFT_IN2, 0)
    pca_set_duty(bus, RIGHT_IN1, duty)
    pca_set_duty(bus, RIGHT_IN2, 0)
    time.sleep(duration)
    motor_stop(bus)
    align_front_wheel()

    # Backword movment: IN1=0, IN2=PWM
def motor_backward(bus, speed_percent=40, duration=1.3):
    duty = int(clamp(speed_percent, 0, 100) * 4095 / 100)
    pca_set_duty(bus, LEFT_IN1, 0)
    pca_set_duty(bus, LEFT_IN2, duty)
    pca_set_duty(bus, RIGHT_IN1, 0)
    pca_set_duty(bus, RIGHT_IN2, duty)
    time.sleep(duration)
    motor_stop(bus)
    align_front_wheel()
# =========================
# (F) Servo movement
# =========================
def align_front_wheel():
    sc.set_angle(SERVO_A, 90)
    time.sleep(0.3)
def servo_pose_hold():
    sc.set_angle(SERVO_C, TILT_ANGLE)
    sc.set_angle(SERVO_D, D_ANGLE)
def servo_pan_move(angle):
    sc.set_angle(SERVO_B, angle)
    time.sleep(SERVO_SETTLE)
    align_front_wheel() 
# =========================
# (H) Camera(webcam) setting
# =========================
# Specify resolution/format as default may be low resolution
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
# MJPG is often said to be advantageous in terms of image quality/frame rate on webcams.
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#HD resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#FPS setting for stable exposure and focus calculations
cam.set(cv2.CAP_PROP_FPS, 30)
#camera stabilization time
time.sleep(1.5)
def capture(tag):
    # Discard warm-up frames to stabilize exposure/focus
    for _ in range(8):
        cam.read()
    ret, frame = cam.read()
    if not ret or frame is None:
        print("[CAM] capture failed")
        return None
    path = f"/home/wolf/{tag}.jpg"
    #saving photos
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return path
# =========================
# (G) Inference execution
# =========================
def infer_from_file(img_path):
    out = subprocess.check_output([INFER_PY, INFER_SCRIPT, img_path], text=True)
    #run AI inference in wolfberry(other python(venv))
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    json_line = lines[-1]
    #tflite-runtime may interleave INFO logs into stdout, implement safe handling so that only the final JSON line is parsed.
    d = json.loads(json_line)
    #transition last line to JSON
    return int(d["pred"]), float(d["conf"])
# =========================
# (H) Modularization of angle-specific inference
# =========================
def infer_at(angle, tag):
#This function makes camera rotate [angle] direction and judge the direction
    servo_pose_hold()
    servo_pan_move(angle)
    img_path = capture(tag)
    if not img_path:
        return None, 0.0
    pred, conf = infer_from_file(img_path)
    return pred, conf
def handle_result(side, pred, conf):
#Decide how the robot will react based on what the AI ​​says
    if pred == WEED_PRED:
        print(f"{side}: Weed Detected (conf={conf:.3f})")
        servo_turn(L_ANGLE)
        servo_turn(R_ANGLE)

        led_on()
        time.sleep(0.4)
        led_off()

    elif pred == STRAW_PRED:
        print(f"{side}: Strawberry Detected (conf={conf:.3f})"

    else:
        print(f"{side}: unknown({pred}) (conf={conf:.3f})")
# =========================
# MAIN LOOP (Step 1~5)
# =========================
bus = None
try:
    bus = SMBus(I2C_BUS)
    # Starting position: servo pose + frontal
    servo_pose_hold()
    servo_pan_move(CENTER)
    while True:
        # Step 1: Stop
        motor_stop(bus)
        time.sleep(STOP_TIME)
        # Step 2: Left Check(-45)
        pred, conf = infer_at(LEFT_ANGLE, "left")
        if pred is not None:
            handle_result("Left", pred, conf)
        servo_pan_move(CENTER)
        # Step 3: Right Check(+45)
        pred, conf = infer_at(RIGHT_ANGLE, "right")
        if pred is not None:
            handle_result("Right", pred, conf)
        servo_pan_move(CENTER)
        # Step 4: Maintain the front and then advance
        servo_pan_move(CENTER)
        motor_forward(bus, MOTOR_SPEED_PERCENT, MOVE_TIME)
        # Step 5: Backword
        BACKWARD_TIME = 1.3 
        motor_backward(bus, MOTOR_SPEED_PERCENT, BACKWARD_TIME)
except KeyboardInterrupt:
    print("\n[STOP] KeyboardInterrupt")
finally:
    #  Safety Device: Immediate Stop + Resource Clearance
    try:
        if bus is not None:
            motor_stop(bus)
    except Exception:
        pass
    try:
        servo_pose_hold()
        servo_pan_move(CENTER) 
    except Exception:
        pass
    try:
        cam.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        if bus is not None:
            bus.close()
    except Exception:
        pass
    print("[CLEANUP] done")
