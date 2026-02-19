# **Trip Data Documentation**

## **Overview**

This document outlines the data structure and attributes used in the **trip data** file `Trip1.json`. The data represents a **synthetic trip** collected from a vehicle's telematics device for **insurance-grade usage**. It includes information about the trip's start and end times, locations, speed, events, and various other attributes necessary for insurance companies to assess risk and driving behavior.

---

## **Data Structure**

The data is structured as a JSON object with the following fields:

### **1. General Trip Information**

| Field Name             | Data Type | Description                                                     |
| ---------------------- | --------- | --------------------------------------------------------------- |
| `trip_id`              | String    | Unique identifier for the trip.                                 |
| `vehicle_id`           | String    | Unique identifier for the vehicle.                              |
| `start_time`           | String    | Timestamp (ISO 8601 format) representing the start of the trip. |
| `end_time`             | String    | Timestamp (ISO 8601 format) representing the end of the trip.   |
| `duration_seconds`     | Integer   | Total duration of the trip in seconds.                          |
| `distance_km`          | Float     | Total distance traveled in kilometers.                          |
| `fuel_consumed_liters` | Float     | Total fuel consumed during the trip in liters.                  |
| `average_speed_kmh`    | Float     | Average speed during the trip in km/h.                          |
| `max_speed_kmh`        | Float     | Maximum speed reached during the trip in km/h.                  |

---

### **2. Location Data**

| Field Name       | Data Type | Description                                                                                                                                        |
| ---------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `start_location` | Object    | GPS coordinates (latitude, longitude) and address where the trip started.                                                                          |
| `end_location`   | Object    | GPS coordinates (latitude, longitude) and address where the trip ended.                                                                            |
| `route`          | Array     | List of GPS coordinates recorded at **1 Hz**, showing the path of the trip. Each element in the array contains latitude, longitude, and timestamp. |

#### **Location Object Structure:**

```json
{
  "lat": Float,
  "lng": Float,
  "address": String
}
```

---

### **3. Speed Data**

| Field Name   | Data Type | Description                                                                                                                                                  |
| ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `speed_data` | Array     | List of speed data points recorded during the trip at **1 Hz** (one per second). Each element contains a timestamp and the speed at that moment in **km/h**. |

#### **Speed Data Object Structure:**

```json
{
  "timestamp": String,  // ISO 8601 formatted timestamp
  "speed_kmh": Float    // Speed in km/h
}
```

---

### **4. Events**

| Field Name | Data Type | Description                                                                                                                                                                                                                          |
| ---------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `events`   | Array     | List of **driving events** detected during the trip, such as **hard braking**, **sharp turns**, and **accelerations**. Each event is recorded with a timestamp, location, and associated G-force threshold that triggered the event. |

#### **Event Object Structure:**

```json
{
  "event_type": String,  // Type of event (e.g., "hard_brake", "sharp_turn", "acceleration")
  "timestamp": String,   // ISO 8601 formatted timestamp
  "location": {          // GPS coordinates (latitude, longitude) where the event occurred
    "lat": Float,
    "lng": Float
  },
  "g_force": Float       // G-force value that triggered the event
}
```

---

### **5. High-Frequency Data (Event-based)**

| Field Name              | Data Type | Description                                                                                                                                                                                                                                         |
| ----------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `high_frequency_events` | Object    | Contains details about events recorded at **higher frequencies** (e.g., during hard braking, sharp turns, or acceleration). These include high-frequency data points such as **speed**, **acceleration**, and **location** recorded in more detail. |

#### **High-Frequency Data Structure for Hard Braking Event:**

```json
{
  "hard_brake_event": {
    "g_force_threshold": Float,  // The G-force that triggered the event
    "data": [                   // List of high-frequency data points recorded during the event
      {
        "timestamp": String,   // ISO 8601 formatted timestamp
        "speed_kmh": Float,    // Speed at that timestamp
        "acceleration": Float, // Acceleration at that timestamp (m/s^2)
        "lat": Float,          // Latitude at that timestamp
        "lng": Float           // Longitude at that timestamp
      }
    ]
  }
}
```

---

### **6. Vehicle Conditions**

| Field Name           | Data Type | Description                                                                                                     |
| -------------------- | --------- | --------------------------------------------------------------------------------------------------------------- |
| `vehicle_conditions` | Object    | Data about the vehicle's **health** during the trip, such as tire pressure, battery level, and oil temperature. |

#### **Vehicle Conditions Object Structure:**

```json
{
  "tire_pressure_front_left": Float,  // Pressure in the front left tire (psi)
  "tire_pressure_front_right": Float, // Pressure in the front right tire (psi)
  "tire_pressure_rear_left": Float,   // Pressure in the rear left tire (psi)
  "tire_pressure_rear_right": Float,  // Pressure in the rear right tire (psi)
  "battery_level_percent": Float,    // Battery level percentage
  "oil_temperature_celsius": Float   // Engine oil temperature in Celsius
}
```

---

### **7. Driver Information**

| Field Name    | Data Type | Description                                                                            |
| ------------- | --------- | -------------------------------------------------------------------------------------- |
| `driver_info` | Object    | Information about the **driver**, including **name**, **age**, and **license status**. |

#### **Driver Information Object Structure:**

```json
{
  "driver_id": String,        // Unique identifier for the driver
  "driver_name": String,      // Full name of the driver
  "driver_age": Integer,      // Age of the driver
  "driver_license_status": String, // Current status of the driver's license (e.g., "Valid")
  "driving_experience_years": Integer // Number of years of driving experience
}
```

---

### **8. Insurance Information**

| Field Name       | Data Type | Description                                                                                          |
| ---------------- | --------- | ---------------------------------------------------------------------------------------------------- |
| `insurance_info` | Object    | Information about the **insurance policy**, including **policy number** and **claimable incidents**. |

#### **Insurance Information Object Structure:**

```json
{
  "policy_number": String,    // Insurance policy number
  "policy_type": String,      // Type of insurance policy (e.g., "Pay-Per-Mile")
  "insurance_company": String, // Name of the insurance company
  "claimable_incidents": Integer // Number of claimable incidents during the trip
}
```

---

## **Sample Data Overview**

The sample **Trip1.json** includes:

* A trip starting at **9:00 AM** and ending at **9:40 AM**.
* **32.8 km** covered with **3.4 liters of fuel** consumed.
* The average speed is **49.2 km/h**, with a maximum speed of **95.7 km/h**.
* Includes **hard braking** events, **sharp turns**, and **acceleration** with **precise high-frequency data** (recorded at 50 Hz during events).
* Vehicle health metrics such as **tire pressure** and **battery level** are included.

---

## **Usage**

This data structure is intended for use by **insurance companies** in **Usage-Based Insurance (UBI)** models, where **driver behavior** (e.g., harsh braking, acceleration) is monitored to calculate premiums based on risk. The data can also be used to improve **telematics systems**, **driver scoring**, and **vehicle maintenance** analysis.

---

## **Conclusion**

This document provides a detailed explanation of the fields used in `Trip1.json`. This data structure ensures accurate, real-time trip tracking and event detection while minimizing transmission costs and ensuring battery efficiency for the telematics device. It is ideal for applications in the insurance industry for **risk assessment**, **claim analysis**, and **driver behavior monitoring**.
