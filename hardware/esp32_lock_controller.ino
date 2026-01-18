/*
 * NASA Medical Inventory - ESP32 Container Lock Controller
 * 
 * This code handles:
 * - WiFi connection
 * - HTTP server for unlock requests
 * - Servo/solenoid control for physical lock
 * - Status LED indicators
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// WiFi credentials - UPDATE THESE
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Pin definitions
const int SERVO_PIN = 18;        // Servo motor for lock mechanism
const int LED_LOCKED = 2;        // LED indicating locked state (red)
const int LED_UNLOCKED = 4;      // LED indicating unlocked state (green)
const int BUZZER_PIN = 5;        // Buzzer for audio feedback
const int BUTTON_PIN = 19;       // Manual lock button

// Lock configuration
const int LOCK_ANGLE = 0;        // Servo angle when locked
const int UNLOCK_ANGLE = 90;     // Servo angle when unlocked
const int UNLOCK_DURATION = 5000; // How long to stay unlocked (ms)

Servo lockServo;
WebServer server(80);
bool isLocked = true;
unsigned long unlockTime = 0;

void setup() {
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(LED_LOCKED, OUTPUT);
  pinMode(LED_UNLOCKED, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize servo
  lockServo.attach(SERVO_PIN);
  lockServo.write(LOCK_ANGLE);
  
  // Set initial LED state
  digitalWrite(LED_LOCKED, HIGH);
  digitalWrite(LED_UNLOCKED, LOW);
  
  // Connect to WiFi
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Setup HTTP server endpoints
  server.on("/", handleRoot);
  server.on("/unlock", HTTP_POST, handleUnlock);
  server.on("/lock", HTTP_POST, handleLock);
  server.on("/status", HTTP_GET, handleStatus);
  server.onNotFound(handleNotFound);
  
  server.begin();
  Serial.println("HTTP server started");
  
  // Startup beep
  beep(100, 2);
}

void loop() {
  server.handleClient();
  
  // Check manual lock button
  if (digitalRead(BUTTON_PIN) == LOW && !isLocked) {
    delay(50); // Debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      lockContainer();
      delay(500);
    }
  }
  
  // Auto-lock after unlock duration
  if (!isLocked && (millis() - unlockTime > UNLOCK_DURATION)) {
    lockContainer();
  }
}

void handleRoot() {
  String html = "<html><head><title>NASA Medical Container</title>";
  html += "<style>body{font-family:Arial;text-align:center;background:#0B3D91;color:white;}";
  html += "h1{color:#FC3D21;}.status{font-size:24px;margin:20px;}</style></head>";
  html += "<body><h1>NASA Medical Container Lock</h1>";
  html += "<div class='status'>Status: " + String(isLocked ? "LOCKED" : "UNLOCKED") + "</div>";
  html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
  html += "<p>Uptime: " + String(millis() / 1000) + " seconds</p>";
  html += "</body></html>";
  
  server.send(200, "text/html", html);
}

void handleUnlock() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    
    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, body);
    
    if (error) {
      server.send(400, "application/json", "{\"error\":\"Invalid JSON\"}");
      return;
    }
    
    const char* astronautId = doc["astronaut_id"];
    const char* timestamp = doc["timestamp"];
    
    Serial.println("Unlock request received");
    Serial.print("Astronaut ID: ");
    Serial.println(astronautId);
    Serial.print("Timestamp: ");
    Serial.println(timestamp);
    
    unlockContainer();
    
    // Send success response
    String response = "{\"success\":true,\"astronaut_id\":\"" + String(astronautId) + 
                     "\",\"status\":\"unlocked\",\"unlock_duration\":" + 
                     String(UNLOCK_DURATION) + "}";
    server.send(200, "application/json", response);
    
  } else {
    server.send(400, "application/json", "{\"error\":\"No data received\"}");
  }
}

void handleLock() {
  lockContainer();
  server.send(200, "application/json", "{\"success\":true,\"status\":\"locked\"}");
}

void handleStatus() {
  String response = "{\"locked\":" + String(isLocked ? "true" : "false") + 
                   ",\"uptime\":" + String(millis() / 1000) + 
                   ",\"ip\":\"" + WiFi.localIP().toString() + "\"}";
  server.send(200, "application/json", response);
}

void handleNotFound() {
  server.send(404, "text/plain", "404: Not Found");
}

void unlockContainer() {
  Serial.println("UNLOCKING CONTAINER");
  
  isLocked = false;
  unlockTime = millis();
  
  // Move servo to unlock position
  lockServo.write(UNLOCK_ANGLE);
  
  // Update LEDs
  digitalWrite(LED_LOCKED, LOW);
  digitalWrite(LED_UNLOCKED, HIGH);
  
  // Audio feedback - unlock sound
  beep(100, 1);
  delay(100);
  beep(200, 1);
  
  Serial.println("Container unlocked for " + String(UNLOCK_DURATION/1000) + " seconds");
}

void lockContainer() {
  Serial.println("LOCKING CONTAINER");
  
  isLocked = true;
  
  // Move servo to lock position
  lockServo.write(LOCK_ANGLE);
  
  // Update LEDs
  digitalWrite(LED_LOCKED, HIGH);
  digitalWrite(LED_UNLOCKED, LOW);
  
  // Audio feedback - lock sound
  beep(200, 1);
  delay(100);
  beep(100, 1);
  
  Serial.println("Container locked");
}

void beep(int duration, int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(duration);
    digitalWrite(BUZZER_PIN, LOW);
    if (i < times - 1) {
      delay(duration);
    }
  }
}