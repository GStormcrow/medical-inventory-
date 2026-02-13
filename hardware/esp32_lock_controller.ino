#include <WiFi.h>
#include <WebServer.h>

// ============================================================================
// CONFIGURATION - UPDATE THESE VALUES
// ============================================================================

// WiFi credentials
const char* ssid = "Daniel's S25+";           // Your WiFi name
const char* password = "sigmarizzlee10";      // Your WiFi password

// Hardware pins
const int LOCK_PIN = 25;                      // Pin connected to relay/solenoid
const int DOOR_SENSOR_PIN = 26;               // Pin connected to door sensor (optional)

// Lock timing
const unsigned long UNLOCK_DURATION = 30000;  // 30 seconds (30000 milliseconds)

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

WebServer server(80);          // Web server on port 80
bool isUnlocked = false;       // Current lock state
unsigned long unlockTime = 0;  // When lock was opened
String lastUser = "";          // Who unlocked it

// ============================================================================
// SETUP - Runs once when ESP32 starts
// ============================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);
 
  // Setup hardware pins
  pinMode(LOCK_PIN, OUTPUT);
  pinMode(DOOR_SENSOR_PIN, INPUT_PULLUP);
  digitalWrite(LOCK_PIN, LOW);  // Start locked
 
  Serial.println("\n=================================");
  Serial.println("NASA Medical Cabinet Lock");
  Serial.println("=================================\n");
 
  // Connect to WiFi
  connectToWiFi();
 
  // Setup web server endpoint (Django will call this)
  server.on("/face-unlock", HTTP_POST, handleFaceUnlock);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/", HTTP_GET, handleRoot);
 
  server.enableCORS(true);  // Allow Django to communicate
  server.begin();
 
  Serial.println("✓ Ready for face recognition!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.println("=================================\n");
}

// ============================================================================
// MAIN LOOP - Runs continuously
// ============================================================================

void loop() {
  server.handleClient();  // Check for commands from Django
 
  // Auto-relock after 30 seconds
  if (isUnlocked && (millis() - unlockTime > UNLOCK_DURATION)) {
    lockCabinet();
  }
 
  // Lock when door opens and closes (optional)
  if (isUnlocked && digitalRead(DOOR_SENSOR_PIN) == HIGH) {
    Serial.println("Door opened");
    delay(3000);  // Wait for medication retrieval
    lockCabinet();
  }
}

// ============================================================================
// FACE UNLOCK HANDLER - Called by Django when face is recognized
// ============================================================================

void handleFaceUnlock() {
  Serial.println("\n👤 Face recognition request received");
 
  // Get astronaut info from Django
  String username = server.hasArg("username") ? server.arg("username") : "Unknown";
  String userId = server.hasArg("user_id") ? server.arg("user_id") : "0";
 
  Serial.print("Astronaut: ");
  Serial.println(username);
 
  // Unlock the cabinet
  unlockCabinet(username);
 
  // Send success response to Django
  String response = "{";
  response += "\"status\":\"unlocked\",";
  response += "\"message\":\"Welcome " + username + "!\",";
  response += "\"unlock_duration\":30";
  response += "}";
 
  server.send(200, "application/json", response);
}

// ============================================================================
// STATUS HANDLER - Check current lock status
// ============================================================================

void handleStatus() {
  String status = isUnlocked ? "unlocked" : "locked";
  int timeRemaining = 0;
 
  if (isUnlocked) {
    unsigned long elapsed = millis() - unlockTime;
    if (elapsed < UNLOCK_DURATION) {
      timeRemaining = (UNLOCK_DURATION - elapsed) / 1000;
    }
  }
 
  String response = "{";
  response += "\"lock\":\"" + status + "\",";
  response += "\"timeRemaining\":" + String(timeRemaining) + ",";
  response += "\"lastUser\":\"" + lastUser + "\"";
  response += "}";
 
  server.send(200, "application/json", response);
}

// ============================================================================
// HOME PAGE - Simple status page
// ============================================================================

void handleRoot() {
  String html = "<!DOCTYPE html><html><head>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<meta http-equiv='refresh' content='3'>";
  html += "<style>";
  html += "body{font-family:Arial;text-align:center;padding:20px;background:#0a0e27;color:white;}";
  html += "h1{color:#3b82f6;} .status{font-size:48px;margin:30px;}";
  html += ".locked{color:#ef4444;} .unlocked{color:#22c55e;}";
  html += "</style></head><body>";
  html += "<h1>Medical Cabinet</h1>";
  html += "<div class='status " + String(isUnlocked ? "unlocked" : "locked") + "'>";
  html += isUnlocked ? "UNLOCKED" : "LOCKED";
  html += "</div>";
 
  if (isUnlocked) {
    unsigned long remaining = (UNLOCK_DURATION - (millis() - unlockTime)) / 1000;
    html += "<p>Auto-lock in: " + String(remaining) + " seconds</p>";
    html += "<p>Last unlocked by: " + lastUser + "</p>";
  }
 
  html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
  html += "</body></html>";
 
  server.send(200, "text/html", html);
}

// ============================================================================
// LOCK CONTROL FUNCTIONS
// ============================================================================

void unlockCabinet(String user) {
  digitalWrite(LOCK_PIN, HIGH);  // Activate solenoid
  isUnlocked = true;
  unlockTime = millis();
  lastUser = user;
 
  Serial.println("UNLOCKED");
  Serial.print("By: ");
  Serial.println(user);
  Serial.println("Auto-lock in 30 seconds\n");
}

void lockCabinet() {
  digitalWrite(LOCK_PIN, LOW);  // Deactivate solenoid
  isUnlocked = false;
 
  Serial.println("LOCKED\n");
}

// ============================================================================
// WIFI CONNECTION
// ============================================================================

void connectToWiFi() {
  WiFi.disconnect(true);
  delay(1000);
  WiFi.mode(WIFI_STA);
  delay(100);
 
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
 
  WiFi.begin(ssid, password);
 
  Serial.print("Connecting");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 60) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
 
  Serial.println();
 
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✓✓✓ WiFi Connected! ✓✓✓");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm\n");
  } else {
    Serial.println("\n✗✗✗ WiFi Failed! ✗✗✗");
    Serial.println("Check SSID and password\n");
    delay(5000);
    ESP.restart();  // Restart and try again
  }
}
