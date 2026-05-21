#include "DHT.h"

#define DHTPIN 2
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

void setup() {

  Serial.begin(115200);

  dht.begin();
}

void loop() {

  delay(2000);

  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (isnan(h) || isnan(t)) {
    Serial.println("Sensor Error");
    return;
  }

  Serial.print(millis());
  Serial.print(",");

  Serial.print(t);
  Serial.print(",");

  Serial.println(h);
}