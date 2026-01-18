from django.core.management.base import BaseCommand
from medical_inventory.models import Astronaut
import face_recognition
import pickle
import os

class Command(BaseCommand):
    help = 'Register astronaut face encoding from an image'

    def add_arguments(self, parser):
        parser.add_argument('astronaut_id', type=int, help='Astronaut database ID')
        parser.add_argument('image_path', type=str, help='Path to astronaut photo')

    def handle(self, *args, **options):
        astronaut_id = options['astronaut_id']
        image_path = options['image_path']
        
        # Check if image exists
        if not os.path.exists(image_path):
            self.stdout.write(self.style.ERROR(f'Image file not found: {image_path}'))
            return
        
        try:
            # Get astronaut
            astronaut = Astronaut.objects.get(id=astronaut_id)
            
            # Load image and create encoding
            self.stdout.write(f'Loading image: {image_path}')
            image = face_recognition.load_image_file(image_path)
            
            # Find face encodings
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) == 0:
                self.stdout.write(self.style.ERROR('No face found in image'))
                return
            elif len(encodings) > 1:
                self.stdout.write(self.style.WARNING(f'Multiple faces found ({len(encodings)}), using first one'))
            
            # Save encoding
            encoding = encodings[0]
            astronaut.face_encoding = pickle.dumps(encoding)
            astronaut.save()
            
            self.stdout.write(self.style.SUCCESS(f'✓ Face encoding registered for {astronaut.name}'))
            
        except Astronaut.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'Astronaut with ID {astronaut_id} not found'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error: {str(e)}'))