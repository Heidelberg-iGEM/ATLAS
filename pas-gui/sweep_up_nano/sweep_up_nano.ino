// Pressure ramp for the SMC electro-pneumatic regulator, driven through an MCP4725 (GY-4725) DAC on an Arduino Nano.
// pas_gui.py sends the step size [V] and the hold time [s] over serial, then logs the output.
#include <Wire.h>

const uint8_t DAC_ADDR = 0x60;     // default address of the GY-4725
const float V_SUPPLY = 5.0;        // Nano supply = DAC reference (GY-4725 VCC -> Nano 5V)
const int LOOPBACK_PIN = A1;       // GY-4725 VOUT -> Nano A1
const int MONITOR_PIN = A0;        // regulator monitor output (pin 4, black) -> Nano A0
const unsigned long INTERVAL_MS = 100;

void setVoltage(float v) {
  v = max(0.0f, min(V_SUPPLY, v));
  uint16_t code = (uint16_t)((v / V_SUPPLY) * 4095);
  uint8_t hi = (code >> 8) & 0x0F;   // upper nibble stays 0 (fast-mode command bits)
  uint8_t lo = code & 0xFF;
  Wire.beginTransmission(DAC_ADDR);
  Wire.write(hi);
  Wire.write(lo);
  Wire.endTransmission();
}

float readLoopback() {
  return analogRead(LOOPBACK_PIN) * V_SUPPLY / 1023.0;
}

float readMonitor() {
  return analogRead(MONITOR_PIN) * V_SUPPLY / 1023.0;
}

String readLine() {
  while (!Serial.available()) {}
  String line = Serial.readStringUntil('\n');
  line.trim();
  return line;
}

void runStep(float v, const char* dir, unsigned long holdMs, unsigned long t0) {
  setVoltage(v);
  unsigned long stepStart = millis();
  unsigned long lastSample = 0;
  while (millis() - stepStart < holdMs) {
    if (millis() - lastSample >= INTERVAL_MS) {
      lastSample = millis();
      Serial.print((millis() - t0) / 1000.0, 2);
      Serial.print(",");
      Serial.print(dir);
      Serial.print(",");
      Serial.print(v, 3);
      Serial.print(",");
      Serial.print(readLoopback(), 3);
      Serial.print(",");
      Serial.println(readMonitor(), 3);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Wire.begin();
  setVoltage(0.0);
}

void loop() {
  Serial.println("Enter step size in V (e.g. 0.1):");
  float stepSize = readLine().toFloat();
  if (stepSize <= 0.0) {
    Serial.println("Invalid step size, starting over.");
    return;
  }

  Serial.println("Enter hold time per step in s (e.g. 3):");
  float holdS = readLine().toFloat();
  if (holdS <= 0.0) {
    Serial.println("Invalid hold time, starting over.");
    return;
  }
  unsigned long holdMs = (unsigned long)(holdS * 1000);

  int numSteps = (int)round(V_SUPPLY / stepSize);

  Serial.print("Sweep 0V->");
  Serial.print(V_SUPPLY);
  Serial.print("V (up only, no return ramp), step ");
  Serial.print(stepSize, 3);
  Serial.print("V, ");
  Serial.print(holdS, 2);
  Serial.println("s each:");
  Serial.println("t_s,dir,setpoint_v,loopback_v,monitor_v");

  unsigned long t0 = millis();

  for (int i = 0; i <= numSteps; i++) {
    float v = min((float)i * stepSize, V_SUPPLY);
    runStep(v, "up", holdMs, t0);
  }

  setVoltage(0.0);
  Serial.println("Sweep up done, 0V set.");
  Serial.println();
}
