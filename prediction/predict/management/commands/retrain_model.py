from django.core.management.base import BaseCommand
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from predict.ml_model import HousePricePredictor


class Command(BaseCommand):
    help = "Retrain the house price prediction model with new data"

    def add_arguments(self, parser):
        parser.add_argument(
            "--samples",
            type=int,
            default=50,
            help="Number of new samples to generate (default: 50)",
        )

    def handle(self, *args, **options):
        num_samples = options["samples"]

        self.stdout.write(
            self.style.SUCCESS(
                f"Starting model retraining with {num_samples} new samples..."
            )
        )

        try:
            # Create predictor instance
            predictor = HousePricePredictor()

            # Retrain the model
            success = predictor.retrain_model(num_samples)

            if success:
                self.stdout.write(
                    self.style.SUCCESS("Model retraining completed successfully!")
                )
                return 0
            else:
                self.stdout.write(self.style.ERROR("Model retraining failed!"))
                return 1

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error during retraining: {str(e)}"))
            return 1
