import joblib
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


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

        # handle categorical features - convert yes/no to 1/0
        for col in available_categorical:
            self.data[col] = self.data[col].map({"yes": 1, "no": 0})

        # handle furnishing status - convert to numerical encoding
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

        # store feature names for later use
        self.feature_names = all_features

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train_scaled, y_train)

        # Save scaler, model, and feature names
        model_dir = os.path.dirname(os.path.abspath(__file__))
        scaler_path = os.path.join(model_dir, "scaler.joblib")
        model_path = os.path.join(model_dir, "ml_model.joblib")
        features_path = os.path.join(model_dir, "feature_names.joblib")

        joblib.dump(scaler, scaler_path)
        joblib.dump(self.model, model_path)
        joblib.dump(self.feature_names, features_path)

        print(f"Model files saved to: {model_dir}")

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
