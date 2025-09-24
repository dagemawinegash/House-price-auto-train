import numpy as np
import pandas as pd
import os


class HousingDataGenerator:
    def __init__(self):
        self.existing_data = None
        self.load_existing_data()

    def load_existing_data(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(project_root, "Housing.csv")

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            self.existing_data = pd.read_csv(dataset_path)
            print(f"Loaded existing data: {self.existing_data.shape}")
            return self.existing_data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def generate_new_data(self, num_samples=50):
        print(f"\nGenerating {num_samples} new housing samples")

        if self.existing_data is None:
            print("Error: No existing data loaded")
            return None

        # Get statistical patterns from existing data
        area_mean = self.existing_data["area"].mean()
        area_std = self.existing_data["area"].std()
        bedrooms_probs = (
            self.existing_data["bedrooms"].value_counts(normalize=True).sort_index()
        )
        bathrooms_probs = (
            self.existing_data["bathrooms"].value_counts(normalize=True).sort_index()
        )
        stories_probs = (
            self.existing_data["stories"].value_counts(normalize=True).sort_index()
        )
        parking_probs = (
            self.existing_data["parking"].value_counts(normalize=True).sort_index()
        )

        # Categorical feature probabilities
        mainroad_prob = (self.existing_data["mainroad"] == "yes").mean()
        guestroom_prob = (self.existing_data["guestroom"] == "yes").mean()
        basement_prob = (self.existing_data["basement"] == "yes").mean()
        hotwater_prob = (self.existing_data["hotwaterheating"] == "yes").mean()
        aircon_prob = (self.existing_data["airconditioning"] == "yes").mean()
        prefarea_prob = (self.existing_data["prefarea"] == "yes").mean()

        # Furnishing status probabilities
        furnishing_probs = self.existing_data["furnishingstatus"].value_counts(
            normalize=True
        )

        # Generate new data
        new_data = []
        for i in range(num_samples):
            # Generate numerical features based on existing patterns
            area = max(1000, np.random.normal(area_mean, area_std))
            bedrooms = np.random.choice(bedrooms_probs.index, p=bedrooms_probs.values)
            bathrooms = np.random.choice(
                bathrooms_probs.index, p=bathrooms_probs.values
            )
            stories = np.random.choice(stories_probs.index, p=stories_probs.values)
            parking = np.random.choice(parking_probs.index, p=parking_probs.values)

            # Generate categorical features
            mainroad = "yes" if np.random.random() < mainroad_prob else "no"
            guestroom = "yes" if np.random.random() < guestroom_prob else "no"
            basement = "yes" if np.random.random() < basement_prob else "no"
            hotwaterheating = "yes" if np.random.random() < hotwater_prob else "no"
            airconditioning = "yes" if np.random.random() < aircon_prob else "no"
            prefarea = "yes" if np.random.random() < prefarea_prob else "no"
            furnishingstatus = np.random.choice(
                furnishing_probs.index, p=furnishing_probs.values
            )

            # Generate price based on existing patterns
            base_price = (
                area * 0.2 + bedrooms * 50000 + bathrooms * 30000 + stories * 20000
            )
            price_variation = np.random.normal(0, base_price * 0.1)
            price = max(100000, base_price + price_variation)

            new_data.append(
                {
                    "price": round(price, 2),
                    "area": round(area, 2),
                    "bedrooms": bedrooms,
                    "bathrooms": bathrooms,
                    "stories": stories,
                    "mainroad": mainroad,
                    "guestroom": guestroom,
                    "basement": basement,
                    "hotwaterheating": hotwaterheating,
                    "airconditioning": airconditioning,
                    "parking": parking,
                    "prefarea": prefarea,
                    "furnishingstatus": furnishingstatus,
                }
            )

        new_df = pd.DataFrame(new_data)
        print(f"Generated {len(new_df)} new samples")
        print(f"New data shape: {new_df.shape}")
        print(
            f"Price range: ${new_df['price'].min():,.0f} - ${new_df['price'].max():,.0f}"
        )

        return new_df

    def save_generated_data(self, new_data, filename="new_housing_data.csv"):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, filename)
            new_data.to_csv(file_path, index=False)
            print(f"Generated data saved to: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving data: {e}")
            return None


if __name__ == "__main__":
    generator = HousingDataGenerator()
    new_data = generator.generate_new_data(10)
    if new_data is not None:
        print("\nSample generated data:")
        print(new_data.head())
        generator.save_generated_data(new_data)
