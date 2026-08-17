// Roda a rede treinada no robô real.
// Gere o policy.h pela interface (botão "Baixar policy.h") ou por:
//   python scripts/export_arduino.py --model models/<seu_modelo>

#include <math.h>
#include "policy.h"

// Pinos dos sensores ultrassônicos (HC-SR04), na MESMA ordem dos sensores da
// simulação: da esquerda para a direita, cobrindo o leque configurado.
const int TRIG_PINS[NUM_SENSORS] = {2, 4, 6, 8};
const int ECHO_PINS[NUM_SENSORS] = {3, 5, 7, 9};

// Ponte H (L298N)
const int MOTOR_PWM[NUM_MOTORS] = {10, 11};
const int MOTOR_IN1[NUM_MOTORS] = {A0, A2};
const int MOTOR_IN2[NUM_MOTORS] = {A1, A3};

// PWM mínimo que faz o motor realmente girar (meça no seu chassi).
const int MIN_DUTY = 40;

float readNormalized(int idx) {
  digitalWrite(TRIG_PINS[idx], LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PINS[idx], HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PINS[idx], LOW);

  // timeout coerente com o alcance: ida e volta a 343 m/s, com folga
  unsigned long timeout = (unsigned long)SENSOR_MAX_RANGE_CM * 60UL;
  unsigned long dur = pulseIn(ECHO_PINS[idx], HIGH, timeout);

  // eco perdido = nada à frente dentro do alcance
  if (dur == 0) return 1.0f;

  float cm = dur * 0.0343f / 2.0f;
  float norm = cm / (float)SENSOR_MAX_RANGE_CM;
  if (norm > 1.0f) norm = 1.0f;
  if (norm < 0.0f) norm = 0.0f;
  return norm;
}

void setMotor(int idx, float pwm) {
  if (fabsf(pwm) < PWM_DEADZONE_FRAC) pwm = 0.0f;

  bool forward = pwm >= 0.0f;
  int duty = (int)(fabsf(pwm) * 255.0f);
  if (duty > 0 && duty < MIN_DUTY) duty = MIN_DUTY;
  if (duty > 255) duty = 255;

  digitalWrite(MOTOR_IN1[idx], forward ? HIGH : LOW);
  digitalWrite(MOTOR_IN2[idx], forward ? LOW : HIGH);
  analogWrite(MOTOR_PWM[idx], duty);
}

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < NUM_SENSORS; i++) {
    pinMode(TRIG_PINS[i], OUTPUT);
    pinMode(ECHO_PINS[i], INPUT);
  }
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(MOTOR_PWM[i], OUTPUT);
    pinMode(MOTOR_IN1[i], OUTPUT);
    pinMode(MOTOR_IN2[i], OUTPUT);
  }
}

void loop() {
  unsigned long t0 = millis();

  float obs[NUM_SENSORS];
  for (int i = 0; i < NUM_SENSORS; i++) obs[i] = readNormalized(i);

  float action[NUM_MOTORS];
  policy_forward(obs, action);

  for (int i = 0; i < NUM_MOTORS; i++) setMotor(i, action[i]);

  for (int i = 0; i < NUM_SENSORS; i++) { Serial.print(obs[i], 2); Serial.print(' '); }
  Serial.print("-> ");
  for (int i = 0; i < NUM_MOTORS; i++) { Serial.print(action[i], 2); Serial.print(' '); }
  Serial.println();

  // mantém o mesmo período de controle usado no treino
  unsigned long elapsed = millis() - t0;
  if (elapsed < CONTROL_PERIOD_MS) delay(CONTROL_PERIOD_MS - elapsed);
}
