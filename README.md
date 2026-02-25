# NASA Medical Inventory System

Advanced medication management system for space missions using facial recognition and Arduino ESP32 for secure locking of medication and automated inventory tracking of medications.

<img width="1919" height="994" alt="Screenshot 2026-02-22 184822" src="https://github.com/user-attachments/assets/1a6b4385-0120-46c3-b230-ffc1fbbedd10" />

## Members
Created by Kowin Ngo, Christian Sang, Daniel Sanjuan, and Jayden Argudo
<br>
Teacher: Emre Gemici
<br>
School: Bergen County Technical High School - Teterboro Campus

## Technology Used
- Django Framework
- C++
- PostgreSQL
- Arduino ESP32 Development Board
- 12V DC Solenoid Lock
- Facial Recognition Python Project (1.3.0)
- Tesseract OCR

## Features

- Facial Recognition Authentication
- Automated Medication Dispensing
- Real-time Inventory Tracking
- Usage Analytics and Reporting
- AI-Powered Pill Recognition
- ESP32-Controlled Smart Lock

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/GStormcrow/medical-inventory-.git
cd medical-inventory-
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Activate virtual environment:
```bash
source venv/bin/activate
```

4. Run migrations:
```bash
python manage.py migrate
```

5. Create superuser:
```bash
python manage.py createsuperuser
```

6. Run development server:
```bash
python manage.py runserver
```

Visit http://localhost:8000 to access the system.

## Project Structure

```
medical-inventory-/
├── manage.py
├── requirements.txt
├── nasa/                    # Main project
├── medical_inventory/       # Core app
├── hardware/               # ESP32 code
├── media/                  # Uploaded files
└── staticfiles/           # Static assets
```

## NASA HUNCH Program

This project is part of the NASA HUNCH (High School Students United with NASA to Create Hardware) program.

## License

MIT License - See LICENSE file for details
