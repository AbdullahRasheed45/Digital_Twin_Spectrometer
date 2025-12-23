"""
Sensor Spectrum Digital Twin - My Implementation
Author: Muhammed
Purpose: ML Research Engineer Interview - Aeris UK

My approach: Build a digital twin that can predict spectral sensor outputs
from experimental conditions using synthetic data + neural networks.
"""

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
    """
    I'm using this to cleanly package the input conditions for each experiment.
    """
    chemical: str       
    temperature: float   
    pressure: float      


@dataclass
class ModelConfig:
    """
    Centralized config - I can change parameters in one place instead of 
    hunting through the code.
    """
    n_wavelengths: int = 256  
    wavelength_range: Tuple[float, float] = (400, 1000) 
    noise_level: float = 0.01 
    train_samples: int = 1000  
    val_samples: int = 200     
    test_samples: int = 200    
    random_seed: int = 42      

    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Helper to create config from dict - useful if I load from JSON"""
        return cls(**config_dict)


class SpectralDataGenerator:
    """
    This is the heart of my synthetic data generation. I'm modeling realistic
    sensor physics so the ML model learns meaningful patterns, not just noise.

    My physics assumptions (based on spectroscopy principles):
    - Each chemical has unique absorption/emission wavelengths (fingerprint)
    - Temperature causes peak shifts (thermal motion) and broadening
    - Pressure causes peak broadening (molecular collisions)
    - Real molecules have multiple peaks (different vibrational/rotational modes)
    - Detectors add noise and have wavelength-dependent sensitivity

    I chose Gaussian peaks because that's what you actually see in spectroscopy
    due to Doppler broadening and other effects.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Create my wavelength axis - evenly spaced points from 400 to 1000 nm
        self.wavelengths = np.linspace(
            config.wavelength_range[0],
            config.wavelength_range[1],
            config.n_wavelengths
        )

        # I'm defining spectral fingerprints for each chemical
        # These are loosely based on real molecular spectroscopy but simplified
        # Each chemical gets:
        #   - Peak positions (where they absorb/emit)
        #   - Relative intensities (some transitions are stronger)
        #   - Base width (natural linewidth)
        #   - Temperature sensitivity (how much peaks shift with temp)
        self.chemical_properties = {
            'H2O': {
                'peaks': [520, 650],           
                'intensities': [1.0, 0.6],     
                'base_width': 35,              
                'temp_sensitivity': 0.3,       
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
        Generate a single synthetic spectrum from given conditions.

        My approach:
        1. Get chemical properties
        2. Calculate temperature and pressure effects
        3. Generate Gaussian peaks at shifted/broadened positions
        4. Add realistic detector (noise, baseline, sensitivity)
        5. Normalize to [0,1] range

        Returns a 256-element array representing the spectrum.
        """
        # First, I need to look up the properties for this chemical
        props = self.chemical_properties.get(conditions.chemical)

        # Start with an empty spectrum - I'll add peaks to this
        spectrum = np.zeros_like(self.wavelengths)

        # Temperature effects - I'm modeling two things here:
        # 1. Systematic shift: hotter molecules 
        # 2. Random fluctuation: thermal jitter in actual measurements
        temp_shift = (conditions.temperature - 25) * props['temp_sensitivity']
        temp_noise = np.random.normal(0, 2)  


        pressure_broadening = conditions.pressure * 1.5


        for peak_center, intensity in zip(props['peaks'], props['intensities']):

            shifted_peak = peak_center + temp_shift + temp_noise


            width = props['base_width'] + pressure_broadening

            # Formula: I(λ) = A * exp(-0.5 * ((λ - λ₀) / σ)²)
            peak_spectrum = intensity * np.exp(
                -0.5 * ((self.wavelengths - shifted_peak) / width) ** 2
            )
            spectrum += peak_spectrum


        # 1. Baseline offset - real detectors have "dark current" 
        baseline = 0.05 * np.random.rand()
        spectrum += baseline
        # 2. Measurement noise - random fluctuations in intensity
        noise = np.random.normal(0, self.config.noise_level, size=spectrum.shape)
        # 3. Detector response - sensitivity varies with wavelength
        detector_response = 1.0 + 0.1 * np.sin(self.wavelengths / 100)
        # Apply detector effects: multiply by response curve, add noise
        spectrum = spectrum * detector_response + noise
        spectrum = np.clip(spectrum, 0, None)
        spectrum = spectrum / (np.max(spectrum) + 1e-8)  

        return spectrum.astype(np.float32)

    def generate_dataset(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a full dataset by randomly sampling conditions and generating spectra.

        I'm sampling uniformly across the parameter space:
        - All chemicals equally likely
        - Temperature uniform in [10, 80]°C
        - Pressure uniform in [0.5, 5.0] bar

        Returns:
            X: Input features (one-hot chemical + temp + pressure)
            y: Output spectra (256 wavelength values each)
        """

        chemicals = list(self.chemical_properties.keys())
        X_conditions = []  
        y_spectra = []     

        for _ in range(n_samples):
            chemical = np.random.choice(chemicals)      
            temperature = np.random.uniform(10, 80)     
            pressure = np.random.uniform(0.5, 5.0)      

            conditions = ExperimentalConditions(chemical, temperature, pressure)
            spectrum = self.generate_spectrum(conditions)

            chemical_onehot = [1 if c == chemical else 0 for c in chemicals]

            x = chemical_onehot + [temperature, pressure]

            X_conditions.append(x)
            y_spectra.append(spectrum)

        return np.array(X_conditions, dtype=np.float32), np.array(y_spectra, dtype=np.float32)


class ModelEvaluator:
    """
    I'm calculating multiple metrics because MSE alone doesn't tell the full story.
    For spectroscopy, peak position is critical - even if overall MSE is low,
    wrong peak position means wrong chemical identification!
    """

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate my evaluation metrics. I chose these because:

        - MSE/RMSE/MAE: Standard regression metrics (overall error)
        - R²: How much variance I'm explaining (0-1 scale, easy to interpret)
        - Peak Error: Critical for spectroscopy - where is the peak?
        - Cosine Similarity: Does the shape match? (ignores magnitude)

        I want multiple perspectives on performance.
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        peak_errors = []
        for true_spec, pred_spec in zip(y_true, y_pred):
            true_peak_idx = np.argmax(true_spec)  
            pred_peak_idx = np.argmax(pred_spec)
            peak_errors.append(abs(true_peak_idx - pred_peak_idx))

        avg_peak_error = np.mean(peak_errors)  
        max_peak_error = np.max(peak_errors)   

        cosine_sims = []
        for true_spec, pred_spec in zip(y_true, y_pred):
            norm_true = np.linalg.norm(true_spec)
            norm_pred = np.linalg.norm(pred_spec)
            if norm_true > 0 and norm_pred > 0: 
                cosine_sim = np.dot(true_spec, pred_spec) / (norm_true * norm_pred)
                cosine_sims.append(cosine_sim)
        avg_cosine_sim = np.mean(cosine_sims)

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(np.sqrt(mse)),  # RMSE has same units as data (easier to interpret)
            'r2': float(r2),
            'avg_peak_error_bins': float(avg_peak_error),
            'max_peak_error_bins': int(max_peak_error),
            'avg_cosine_similarity': float(avg_cosine_sim),
        }

    @staticmethod
    def print_metrics(metrics: Dict, model_name: str = "Model"):
        """Pretty print results - I want clean, readable output for my experiments"""
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
    """
    This is my main pipeline that ties everything together.
    I wanted a clean API where I can just call prepare_data() → train_model() → evaluate()
    without worrying about the details.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.generator = SpectralDataGenerator(config)  
        self.scaler_X = StandardScaler()  
        self.model = None  
        self.data = {}   

    def prepare_data(self, verbose: bool = True):
        """
        Generate and preprocess all my datasets (train/val/test).

        Important: I use different random seeds for each split to ensure
        they're independent. This prevents data leakage!
        """
        if verbose:
            print("Generating synthetic datasets...")

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

        # Feature scaling -  for neural networks!
     
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
            print(f"Data generated:")
            print(f"  Training samples:   {X_train.shape[0]}")
            print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Test samples:       {X_test.shape[0]}")
            print(f"  Input features:     {X_train.shape[1]}")
            print(f"  Output dimensions:  {y_train.shape[1]}")

    def train_model(self, model_type: str = 'mlp', verbose: bool = True):
        """
        Train either MLP (neural network) or Random Forest.

        I'm testing both to show the NN is actually better - always good
        to have a baseline comparison!
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
            print(f" Training completed in {train_time:.2f}s")

        return train_time

    def evaluate(self, split: str = 'test', verbose: bool = True) -> Dict:
        """
        Evaluate my model on any split (train/val/test).

        I use this to check:
        - Training set: Am I learning? (should have low error)
        - Validation set: Am I overfitting? (should be similar to train)
        - Test set: How will I perform on new data? (final evaluation)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        # Get the data for requested split
        X = self.data[f'X_{split}']
        y_true = self.data[f'y_{split}']

        # Make predictions
        y_pred = self.model.predict(X)

        # Calculate all my metrics
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)

        if verbose:
            ModelEvaluator.print_metrics(metrics, f"Model ({split.upper()} set)")

        return metrics, y_pred

    def predict(self, conditions: ExperimentalConditions) -> np.ndarray:
        """
        Predict spectrum for new experimental conditions.
        This is the "inference" mode - what I'd use in production.
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        # Need to encode the conditions same way as training data
        chemicals = ['H2O', 'CO2', 'CH4', 'N2', 'O2']
        chemical_onehot = [1 if c == conditions.chemical else 0 for c in chemicals]
        x = np.array([chemical_onehot + [conditions.temperature, conditions.pressure]])

        # IMPORTANT: Must scale using the same scaler from training!
        x_scaled = self.scaler_X.transform(x)

        # Predict and extract the spectrum (returns 2D array, I want 1D)
        spectrum = self.model.predict(x_scaled)[0]

        return spectrum


def main():
    """
    Main execution - ties everything together.
    I set this up with argparse so I can easily experiment with different
    configurations without editing the code.
    """
    # Set up command line arguments for flexibility
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

    config = ModelConfig(
        n_wavelengths=args.wavelengths,
        train_samples=args.samples,
        val_samples=args.samples // 5, 
        test_samples=args.samples // 5,
        random_seed=args.seed
    )



    print(f"Configuration:")
    print(f"  Wavelength bins: {config.n_wavelengths}")
    print(f"  Training samples: {config.train_samples}")
    print(f"  Model type: {args.model.upper()}")
    print("="*70)

    pipeline = DigitalTwinPipeline(config)
    pipeline.prepare_data()  
    pipeline.train_model(model_type=args.model) 
    test_metrics, _ = pipeline.evaluate(split='test')
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