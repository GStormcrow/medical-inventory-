# NASA Medical Inventory System

Advanced medication management system for space missions using facial recognition and automated inventory tracking.

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

## Documentation

For detailed setup instructions, see the installation guide in the docs.

## NASA HUNCH Program

This project is part of the NASA HUNCH (High School Students United with NASA to Create Hardware) program.

## License

MIT License - See LICENSE file for details
