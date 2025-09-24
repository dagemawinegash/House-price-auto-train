import joblib
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_generator import HousingDataGenerator


class HousePricePredictor:
    def load_dataset(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            dataset_path = os.path.join(project_root, "Housing.csv")

            print(f"Looking for dataset at: {dataset_path}")

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found at {dataset_path}")

            data = pd.read_csv(dataset_path)
            print(f"Dataset loaded successfully. Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")

            return data

        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "area": np.random.uniform(1000, 5000, 1000),
                "bedrooms": np.random.randint(1, 6, 1000),
                "bathrooms": np.random.randint(1, 4, 1000),
                "stories": np.random.randint(1, 4, 1000),
                "mainroad": np.random.choice(["yes", "no"], 1000),
                "guestroom": np.random.choice(["yes", "no"], 1000),
                "basement": np.random.choice(["yes", "no"], 1000),
                "hotwaterheating": np.random.choice(["yes", "no"], 1000),
                "airconditioning": np.random.choice(["yes", "no"], 1000),
                "price": None,
            }
        )

        data["price"] = (
            data["area"] * 0.2
            + data["bedrooms"] * 50000
            + data["bathrooms"] * 30000
            + data["stories"] * 20000
            + np.random.normal(0, 50000, 1000)
        )

        return data

    def init(self):
        self.data = self.load_dataset()
        self._train_model()

    def predict(
        self,
        area,
        bedrooms,
        bathrooms,
        stories,
        parking,
        mainroad,
        guestroom,
        basement,
        hotwaterheating,
        airconditioning,
        prefarea,
        furnishingstatus,
    ):
        # load model files from the same directory as this script
        model_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        model_path = os.path.join(model_dir, "ml_model.joblib")
        features_path = os.path.join(model_dir, "feature_names.joblib")

        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)

        # convert categorical inputs to 1/0
        mainroad = 1 if mainroad.lower() == "yes" else 0
        guestroom = 1 if guestroom.lower() == "yes" else 0
        basement = 1 if basement.lower() == "yes" else 0
        hotwaterheating = 1 if hotwaterheating.lower() == "yes" else 0
        airconditioning = 1 if airconditioning.lower() == "yes" else 0
        prefarea = 1 if prefarea.lower() == "yes" else 0

        # convert furnishing status to numerical
        furnishing_mapping = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
        furnishingstatus = furnishing_mapping.get(furnishingstatus.lower(), 0)

        # create input data in the same order as training
        input_data = np.array(
            [
                [
                    area,
                    bedrooms,
                    bathrooms,
                    stories,
                    parking,
                    mainroad,
                    guestroom,
                    basement,
                    hotwaterheating,
                    airconditioning,
                    prefarea,
                    furnishingstatus,
                ]
            ]
        )

        # create DataFrame with proper column names to avoid warning
        input_df = pd.DataFrame(input_data, columns=feature_names)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        return round(prediction, 2)

    def retrain_model(self, num_new_samples=50):
        print(f"Retraining model with {num_new_samples} new samples...")

        try:
            # load existing data
            existing_data = self.load_dataset()
            print(f"Loaded existing data: {existing_data.shape}")

            # generate new data
            generator = HousingDataGenerator()
            new_data = generator.generate_new_data(num_new_samples)

            if new_data is None:
                print("Error: Failed to generate new data")
                return False

            # combine existing and new data
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            print(f"Combined dataset shape: {combined_data.shape}")

            # set the combined data and retrain
            self.data = combined_data
            self._train_model()

            print(f"Model retrained successfully!")
            return True

        except Exception as e:
            print(f"Error during retraining: {e}")
            return False

    def _train_model(self):
        numerical_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]
        categorical_features = [
            "mainroad",
            "guestroom",
            "basement",
            "hotwaterheating",
            "airconditioning",
            "prefarea",
        ]
        furnishing_features = ["furnishingstatus"]

        available_numerical = [
            col for col in numerical_features if col in self.data.columns
        ]
        available_categorical = [
            col for col in categorical_features if col in self.data.columns
        ]
        available_furnishing = [
            col for col in furnishing_features if col in self.data.columns
        ]

        print(f"Using numerical features: {available_numerical}")
        print(f"Using categorical features: {available_categorical}")
        print(f"Using furnishing features: {available_furnishing}")

        # handle categorical features
        for col in available_categorical:
            self.data[col] = self.data[col].map({"yes": 1, "no": 0})

        # handle furnishing status
        for col in available_furnishing:
            self.data[col] = self.data[col].map(
                {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
            )

        # combine all features
        all_features = (
            available_numerical + available_categorical + available_furnishing
        )
        print(f"All features: {all_features}")

        print(f"Missing values: {self.data[all_features].isnull().sum()}")

        # remove rows with missing values
        self.data = self.data.dropna(subset=all_features + ["price"])
        print(f"Dataset shape after removing missing values: {self.data.shape}")

        X = self.data[all_features]
        y = self.data["price"]

        # tore feature names
        self.feature_names = all_features

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Cross-validation
        print("Performing cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_rmse = np.sqrt(-cv_scores)
        print(
            f"Cross-validation RMSE: {cv_rmse.mean():.2f} (+/- {cv_rmse.std() * 2:.2f})"
        )

        # Test set performance
        X_test_scaled = scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"Test set RMSE: {test_rmse:.2f}")

        # Save model files
        model_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        model_path = os.path.join(model_dir, "ml_model.joblib")
        features_path = os.path.join(model_dir, "feature_names.joblib")

        joblib.dump(scaler, scaler_path)
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_names, features_path)

        print(f"Model files saved to: {model_dir}")


# only train the model if the model files don't exist
model_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_dir, "ml_model.joblib")
scaler_path = os.path.join(model_dir, "scaler.joblib")
features_path = os.path.join(model_dir, "feature_names.joblib")

if not (
    os.path.exists(model_path)
    and os.path.exists(scaler_path)
    and os.path.exists(features_path)
):
    print("Model files not found. Training new model...")
    HousePricePredictor().init()
else:
    print("Model files found. Skipping training.")
