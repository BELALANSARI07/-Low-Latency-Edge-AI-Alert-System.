# MQTT Broker Security & Hardening Guide

This document outlines the security procedures for hardening the **Eclipse Mosquitto** MQTT broker in production deployments of the **Low-Latency Edge AI Alert System**.

---

## 1. Threat Model & Risks

In unattended edge environments (e.g., highway poles, industrial corridors, remote IoT nodes), unauthenticated or unencrypted MQTT brokers expose critical risks:
- **Eavesdropping**: Video telemetry, alert locations, and system status intercepted over unencrypted TCP.
- **Spoofing**: Unauthorized publishers injecting fraudulent hazard alerts or triggering GPIO actuators.
- **Denial of Service**: Unauthenticated clients exhausting broker connections and delaying alert distribution.

---

## 2. Authentication Configuration

In production (`mosquitto.prod.conf`), anonymous connections are strictly prohibited:

```conf
allow_anonymous false
password_file /mosquitto/config/passwd
```

### Creating Credentials with `mosquitto_passwd`

Generate an encrypted password file:

```bash
# Create new password file with edge publisher user
mosquitto_passwd -c mosquitto/config/passwd edge_publisher

# Add additional subscriber users (e.g. backend dashboard)
mosquitto_passwd mosquitto/config/passwd dashboard_subscriber
```

---

## 3. Topic Authorization (ACLs)

Define an Access Control List file at `mosquitto/config/acl.conf` to enforce least-privilege topic isolation:

```conf
# Edge Camera Node (Can publish alerts and telemetry, but cannot publish to actuator control)
user edge_publisher
topic write alert/system/01
topic write telemetry/edge/01

# Monitoring Dashboard / Sirens (Can read alerts and telemetry)
user dashboard_subscriber
topic read alert/system/+
topic read telemetry/edge/+
```

---

## 4. TLS Encryption (Port 8883)

To encrypt traffic against network sniffing and MITM attacks:

### Step 1: Generate Self-Signed Certificates (or use CA)

```bash
# Generate CA Private Key and Certificate
openssl req -new -x509 -days 3650 -extensions v3_ca -keyout ca.key -out ca.crt

# Generate Server Key and CSR
openssl genrsa -out server.key 2048
openssl req -new -out server.csr -key server.key

# Sign Server Certificate
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365
```

### Step 2: Configure Broker TLS Listener

In `mosquitto.prod.conf`:
```conf
listener 8883
cafile /mosquitto/config/certs/ca.crt
certfile /mosquitto/config/certs/server.crt
keyfile /mosquitto/config/certs/server.key
tls_version tlsv1.3
```

---

## 5. Application `.env` Configuration

Configure the edge application to use secure authenticated MQTT:

```ini
MQTT_BROKER_HOST=mqtt.edge-internal.lan
MQTT_BROKER_PORT=8883
MQTT_USERNAME=edge_publisher
MQTT_PASSWORD=SuperSecretProductionPassword!
MQTT_USE_TLS=true
MQTT_CA_CERT=/etc/ssl/certs/ca.crt
MQTT_ALERT_TOPIC=alert/system/01
MQTT_TELEMETRY_TOPIC=telemetry/edge/01
MQTT_QOS=1
```
