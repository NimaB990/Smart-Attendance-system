# 🎓 Smart Attendance System  

An intelligent attendance management system that combines **RFID authentication** with **AI-powered face recognition** using Python and OpenCV. The system securely logs attendance in a **cloud database**, prevents proxy attendance, and captures unauthorized attempts.  

---

## ✨ Key Features  

- 🔑 **RFID Card Validation** – Every student must first scan their RFID card.  
- 👤 **Face Recognition** – Real-time webcam face verification using `face_recognition` and OpenCV.  
- ⏱️ **Automated Attendance Logging** – Saves student ID, date, and time into a cloud-hosted database.  
- 📸 **Unauthorized Detection** – Captures and stores snapshots of failed or invalid attempts.  
- ☁️ **Cloud Integration** – Attendance records securely stored for easy access and analysis.  

---

## 🛠️ Tech Stack  

- **Language**: Python  
- **Libraries**: OpenCV, face_recognition, NumPy, Pillow, dotenv, tkinter  
- **Database**: MySQL / Supabase (Cloud-hosted)  
- **Hardware**: USB RFID Reader, Webcam 
