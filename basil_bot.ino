#include <ESP32Servo.h>

const int NUM_SERVOS = 5;
const int servoPins[NUM_SERVOS] = {1, 2, 3, 4, 5};  // Physical pin mapping

// Center/start angles
const int centerAngles[NUM_SERVOS] = {40, 100, 40, 60, 10};
// Angle limits per servo
const int angleLimits[NUM_SERVOS][2] = {
  {0, 180}, {10, 170}, {10, 130}, {0, 180}, {0, 180}
};

// Servo role mapping for clarity
#define BASE_SERVO     0  // Left/right
#define SHOULDER_SERVO 1  // Up/down
#define ELBOW_SERVO    2  // Forward extension
#define WRIST_SERVO    3  // Wrist pitch
#define CUTTER_SERVO   4  // Cutter action

Servo servos[NUM_SERVOS];
int servoAngles[NUM_SERVOS] = {40, 55, 120, 90, 70};

unsigned long lastMoveTime = 0;
unsigned long moveCooldown = 250;

void moveSmooth(int index, int targetAngle, int stepDelay = 5) {
  int current = servoAngles[index];
  int step = (current < targetAngle) ? 1 : -1;

  for (int angle = current; angle != targetAngle; angle += step) {
    servos[index].write(angle);
    delay(stepDelay);
  }

  servos[index].write(targetAngle);
  servoAngles[index] = targetAngle;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(servoPins[i], 500, 2400);
    servos[i].write(centerAngles[i]);
  }

  Serial.println("✅ ESP32 Ready!");
}

void loop() {
  if (Serial.available() > 0) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
    Serial.println("🛰️ Received: " + msg);

    if (msg == "FORWARD_STEP") {
      moveSmooth(SHOULDER_SERVO, servoAngles[SHOULDER_SERVO] + 3);
      moveSmooth(ELBOW_SERVO, servoAngles[ELBOW_SERVO] + 3);
      Serial.println("✅ Forward step.");
      return;
    }

    if (msg == "CUT") {
      moveSmooth(CUTTER_SERVO, 150);  // extend cutter
      delay(300);
      moveSmooth(CUTTER_SERVO, centerAngles[CUTTER_SERVO]);  // reset
      Serial.println("✂️ Cut performed.");
      return;
    }

    if (millis() - lastMoveTime < moveCooldown) {
      Serial.println("⏳ Cooldown active.");
      return;
    }

    int c1 = msg.indexOf(',');
    int c2 = msg.indexOf(',', c1 + 1);
    if (c1 > 0 && c2 > c1) {
      float x = msg.substring(0, c1).toFloat();
      float y = msg.substring(c1 + 1, c2).toFloat();

      float dx = x - 0.5;
      float dy = y - 0.5;
      float deadZone = 0.05;

      int stepSizeX = (abs(dx) > 0.2) ? 3 : (abs(dx) > 0.1 ? 2 : 1);
      int stepSizeY = (abs(dy) > 0.2) ? 3 : (abs(dy) > 0.1 ? 2 : 1);

      bool moved = false;
      if (abs(dx) > deadZone) {
        int delta = (dx > 0) ? stepSizeX : -stepSizeX;
        moveSmooth(BASE_SERVO, constrain(servoAngles[BASE_SERVO] + delta, angleLimits[BASE_SERVO][0], angleLimits[BASE_SERVO][1]));
        moved = true;
      }

      if (abs(dy) > deadZone) {
        int delta = (dy < 0) ? -stepSizeY : stepSizeY;
        moveSmooth(SHOULDER_SERVO, constrain(servoAngles[SHOULDER_SERVO] + delta, angleLimits[SHOULDER_SERVO][0], angleLimits[SHOULDER_SERVO][1]));
        moved = true;
      }

      if (moved) lastMoveTime = millis();
    }
  }

  delay(10);
}
