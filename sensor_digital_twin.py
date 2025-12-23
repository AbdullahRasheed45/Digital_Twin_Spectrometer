import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time 



@dataclass
class ExperimentalConditions:
    """Represents experimental conditions for sensor measurement"""
    chemical: str
    temperature: float  
    pressure: float     


@dataclass
class ModelConfig:
    """Configuration for model training and data generation"""
    n_wavelengths: int = 256
    wavelength_range: Tuple[float, float] = (400, 1000)
    noise_level: float = 0.01
    train_samples: int = 1000
    val_samples: int = 200
    test_samples: int = 200
    random_seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        return cls(**config_dict)


class SpectralDataGenerator:
    """
    Generates synthetic spectral data mimicking realistic sensor behavior.

    Physical Assumptions:
    ---------------------
    1. Each chemical has characteristic absorption/emission peaks
    2. Temperature shifts peak position due to:
       - Thermal broadening (Doppler effect)
       - Energy level population changes
    3. Pressure affects peak width (collision broadening)
    4. Multiple peaks exist for complex molecules (vibrational modes)
    5. Background noise represents electronic/optical detector noise
    6. Detector response varies with wavelength

    The synthetic data is designed to be physically plausible while
    remaining computationally tractable for ML training.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.wavelengths = np.linspace(
            config.wavelength_range[0],
            config.wavelength_range[1],
            config.n_wavelengths
        )

        # Define chemical properties with multiple peaks
        self.chemical_properties = {
            'H2O': {
                'peaks': [520, 650],
                'intensities': [1.0, 0.6],
                'base_width': 35,
                'temp_sensitivity': 0.3,  # nm/°C
            },
            'CO2': {
                'peaks': [700, 850],
                'intensities': [1.0, 0.4],
                'base_width': 40,
                'temp_sensitivity': 0.5,
            },
            'CH4': {
                'peaks': [900, 750],
                'intensities': [1.0, 0.5],
                'base_width': 30,
                'temp_sensitivity': 0.4,
            },
            'N2': {
                'peaks': [600, 780],
                'intensities': [0.8, 0.9],
                'base_width': 45,
                'temp_sensitivity': 0.25,
            },
            'O2': {
                'peaks': [630, 760],
                'intensities': [1.0, 0.7],
                'base_width': 38,
                'temp_sensitivity': 0.35,
            }
        }

    def generate_spectrum(self, conditions: ExperimentalConditions) -> np.ndarray:
        """
        Generate a synthetic spectrum based on experimental conditions.

        Args:
            conditions: ExperimentalConditions object

        Returns:
            spectrum: numpy array of normalized intensity values [0, 1]
        """
        props = self.chemical_properties.get(conditions.chemical)
        if props is None:
            raise ValueError(f"Unknown chemical: {conditions.chemical}")

        spectrum = np.zeros_like(self.wavelengths)

        # Temperature effect: shift peak positions + thermal fluctuation
        temp_shift = (conditions.temperature - 25) * props['temp_sensitivity']
        temp_noise = np.random.normal(0, 2)  # Random thermal fluctuations

        # Pressure effect: broaden peaks (collision broadening)
        pressure_broadening = conditions.pressure * 1.5

        # Generate multiple peaks for each chemical
        for peak_center, intensity in zip(props['peaks'], props['intensities']):
            shifted_peak = peak_center + temp_shift + temp_noise
            width = props['base_width'] + pressure_broadening

            # Gaussian peak with physical effects
            peak_spectrum = intensity * np.exp(
                -0.5 * ((self.wavelengths - shifted_peak) / width) ** 2
            )
            spectrum += peak_spectrum

        # Add baseline offset (detector dark current)
        baseline = 0.05 * np.random.rand()
        spectrum += baseline

        # Add wavelength-dependent noise (detector sensitivity varies)
        noise = np.random.normal(0, self.config.noise_level, size=spectrum.shape)
        detector_response = 1.0 + 0.1 * np.sin(self.wavelengths / 100)

        spectrum = spectrum * detector_response + noise

        # Normalize to [0, 1] range
        spectrum = np.clip(spectrum, 0, None)
        spectrum = spectrum / (np.max(spectrum) + 1e-8)

        return spectrum.astype(np.float32)

    def generate_dataset(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a dataset of spectra with varying conditions.

        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility

        Returns:
            X: Input conditions (n_samples, n_features)
            y: Output spectra (n_samples, n_wavelengths)
        """
        if seed is not None:
            np.random.seed(seed)

        chemicals = list(self.chemical_properties.keys())
        X_conditions = []
        y_spectra = []

        for _ in range(n_samples):
            # Sample conditions uniformly
            chemical = np.random.choice(chemicals)
            temperature = np.random.uniform(10, 80)  # 10-80°C
            pressure = np.random.uniform(0.5, 5.0)   # 0.5-5.0 bar

            conditions = ExperimentalConditions(chemical, temperature, pressure)
            spectrum = self.generate_spectrum(conditions)

            # Encode conditions: one-hot for chemical + continuous for temp/pressure
            chemical_onehot = [1 if c == chemical else 0 for c in chemicals]
            x = chemical_onehot + [temperature, pressure]

            X_conditions.append(x)
            y_spectra.append(spectrum)

        return np.array(X_conditions, dtype=np.float32), np.array(y_spectra, dtype=np.float32)


class ModelEvaluator:
    """Comprehensive model evaluation with domain-specific metrics"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate evaluation metrics tailored for spectral prediction.

        Metrics:
        --------
        - MSE/RMSE: Overall reconstruction error
        - MAE: Average deviation per wavelength
        - R²: Variance explained by the model
        - Peak Error: Error in peak position (critical for spectroscopy)
        - Cosine Similarity: Spectral shape matching quality

        Args:
            y_true: Ground truth spectra
            y_pred: Predicted spectra

        Returns:
            Dictionary of metric values
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate peak position errors (critical for spectroscopy)
        peak_errors = []
        for true_spec, pred_spec in zip(y_true, y_pred):
            true_peak_idx = np.argmax(true_spec)
            pred_peak_idx = np.argmax(pred_spec)
            peak_errors.append(abs(true_peak_idx - pred_peak_idx))
        avg_peak_error = np.mean(peak_errors)
        max_peak_error = np.max(peak_errors)

        # Cosine similarity (measures shape matching)
        cosine_sims = []
        for true_spec, pred_spec in zip(y_true, y_pred):
            norm_true = np.linalg.norm(true_spec)
            norm_pred = np.linalg.norm(pred_spec)
            if norm_true > 0 and norm_pred > 0:
                cosine_sim = np.dot(true_spec, pred_spec) / (norm_true * norm_pred)
                cosine_sims.append(cosine_sim)
        avg_cosine_sim = np.mean(cosine_sims)

        return {
            'mse': float(mse),  # Convert to Python float
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),
            'r2': float(r2),
            'avg_peak_error_bins': float(avg_peak_error),
            'max_peak_error_bins': int(max_peak_error),  # Convert to Python int
            'avg_cosine_similarity': float(avg_cosine_sim),
        }

    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "Model"):
        """Pretty print evaluation metrics"""
        print(f"\n{'='*70}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*70}")
        print(f"MSE:                      {metrics['mse']:.6f}")
        print(f"RMSE:                     {metrics['rmse']:.6f}")
        print(f"MAE:                      {metrics['mae']:.6f}")
        print(f"R² Score:                 {metrics['r2']:.4f}")
        print(f"Avg Peak Error (bins):    {metrics['avg_peak_error_bins']:.2f}")
        print(f"Max Peak Error (bins):    {metrics['max_peak_error_bins']:.0f}")
        print(f"Avg Cosine Similarity:    {metrics['avg_cosine_similarity']:.4f}")
        print(f"{'='*70}\n")


class DigitalTwinPipeline:
    """Complete end-to-end pipeline for sensor digital twin"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.generator = SpectralDataGenerator(config)
        self.scaler_X = StandardScaler()
        self.model = None
        self.data = {}

    def prepare_data(self, verbose: bool = True):
        """Generate and prepare datasets"""
        if verbose:
            print("Generating synthetic datasets...")

        # Generate datasets with different seeds for independence
        X_train, y_train = self.generator.generate_dataset(
            self.config.train_samples,
            seed=self.config.random_seed
        )
        X_val, y_val = self.generator.generate_dataset(
            self.config.val_samples,
            seed=self.config.random_seed + 1
        )
        X_test, y_test = self.generator.generate_dataset(
            self.config.test_samples,
            seed=self.config.random_seed + 2
        )

        # Standardize input features (critical for neural networks)
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)

        self.data = {
            'X_train': X_train_scaled, 'y_train': y_train,
            'X_val': X_val_scaled, 'y_val': y_val,
            'X_test': X_test_scaled, 'y_test': y_test,
            'X_train_raw': X_train,
            'X_val_raw': X_val,
            'X_test_raw': X_test
        }

        if verbose:
            print(f"✓ Data generated:")
            print(f"  Training samples:   {X_train.shape[0]}")
            print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Test samples:       {X_test.shape[0]}")
            print(f"  Input features:     {X_train.shape[1]}")
            print(f"  Output dimensions:  {y_train.shape[1]}")

    def train_model(self, model_type: str = 'mlp', verbose: bool = True):
        """
        Train a model on the prepared data.

        Args:
            model_type: 'mlp' or 'rf' (random forest)
            verbose: Print training progress
        """
        if verbose:
            print(f"\nTraining {model_type.upper()} model...")

        start = time.time()

        if model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 256, 256, 128),
                activation='relu',
                solver='adam',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=self.config.random_seed,
                verbose=False
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=30,
                random_state=self.config.random_seed,
                n_jobs=-1,
                verbose=0
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.fit(self.data['X_train'], self.data['y_train'])
        train_time = time.time() - start

        if verbose:
            print(f"Training completed in {train_time:.2f}s")

        return train_time

    def evaluate(self, split: str = 'test', verbose: bool = True) -> Dict:
        """
        Evaluate model on specified data split.

        Args:
            split: 'train', 'val', or 'test'
            verbose: Print evaluation results

        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        X = self.data[f'X_{split}']
        y_true = self.data[f'y_{split}']

        y_pred = self.model.predict(X)
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)

        if verbose:
            ModelEvaluator.print_metrics(metrics, f"Model ({split.upper()} set)")

        return metrics, y_pred

    def predict(self, conditions: ExperimentalConditions) -> np.ndarray:
        """
        Predict spectrum for given experimental conditions.

        Args:
            conditions: ExperimentalConditions object

        Returns:
            Predicted spectrum as numpy array
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        # Encode conditions
        chemicals = ['H2O', 'CO2', 'CH4', 'N2', 'O2']
        chemical_onehot = [1 if c == conditions.chemical else 0 for c in chemicals]
        x = np.array([chemical_onehot + [conditions.temperature, conditions.pressure]])

        # Scale and predict
        x_scaled = self.scaler_X.transform(x)
        spectrum = self.model.predict(x_scaled)[0]

        return spectrum

    def show_sample_predictions(self, n_samples: int = 5):
        """Display sample predictions with their conditions"""
        _, y_pred = self.evaluate(split='test', verbose=False)

        print(f"\nSample Predictions (first {n_samples}):")
        print("=" * 70)

        chemicals = ['H2O', 'CO2', 'CH4', 'N2', 'O2']

        for i in range(min(n_samples, len(y_pred))):
            true_spec = self.data['y_test'][i]
            pred_spec = y_pred[i]
            conditions = self.data['X_test_raw'][i]

            # Decode conditions
            chem_idx = np.argmax(conditions[:5])
            chemical = chemicals[chem_idx]
            temp = conditions[5]
            pressure = conditions[6]

            true_peak = np.argmax(true_spec)
            pred_peak = np.argmax(pred_spec)
            spec_mse = mean_squared_error(true_spec.reshape(1, -1), pred_spec.reshape(1, -1))

            print(f"\nSample {i+1}:")
            print(f"  Conditions: {chemical}, T={temp:.1f}°C, P={pressure:.2f} bar")
            print(f"  True peak: bin {true_peak}, Predicted peak: bin {pred_peak}")
            print(f"  Peak error: {abs(true_peak - pred_peak)} bins")
            print(f"  Spectrum MSE: {spec_mse:.6f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Sensor Digital Twin - Spectral Prediction System'
    )
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of training samples')
    parser.add_argument('--wavelengths', type=int, default=256,
                       help='Number of wavelength bins')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'rf'],
                       help='Model type: mlp or rf')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Configuration
    config = ModelConfig(
        n_wavelengths=args.wavelengths,
        train_samples=args.samples,
        val_samples=args.samples // 5,
        test_samples=args.samples // 5,
        random_seed=args.seed
    )

    print("="*70)
    print("SENSOR DIGITAL TWIN - SPECTRAL PREDICTION SYSTEM")
    print("="*70)
    print(f"Configuration:")
    print(f"  Wavelength bins: {config.n_wavelengths}")
    print(f"  Training samples: {config.train_samples}")
    print(f"  Model type: {args.model.upper()}")
    print("="*70)

    # Run pipeline
    pipeline = DigitalTwinPipeline(config)
    pipeline.prepare_data()
    pipeline.train_model(model_type=args.model)

    # Evaluate
    test_metrics, _ = pipeline.evaluate(split='test')

    # Show sample predictions
    pipeline.show_sample_predictions(n_samples=5)

    # Save results
    results = {
        'config': {
            'n_wavelengths': config.n_wavelengths,
            'train_samples': config.train_samples,
            'model_type': args.model,
        },
        'metrics': test_metrics
    }

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("Results saved to results.json")
    print("="*70)


if __name__ == "__main__":
    main()