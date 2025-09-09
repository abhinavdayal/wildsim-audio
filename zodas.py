import numpy as np
import scipy.signal
import scipy.fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ZODASPipeline:
    """
    ZODAS Signal Processing Pipeline for Elephant Sound Detection
    Based on the ReSpeaker USB 4-mic array geometry
    """
    
    def __init__(self, sample_rate: int = 16000, frame_size: int = 512, overlap: float = 0.5):
        # ReSpeaker 4-mic array geometry (in meters)
        self.mic_positions = np.array([
            [-0.032, 0.0, 0.0],    # Mic 1 (Left)
            [0.0, -0.032, 0.0],    # Mic 2 (Back) 
            [0.032, 0.0, 0.0],     # Mic 3 (Right)
            [0.0, 0.032, 0.0]      # Mic 4 (Front)
        ])
        
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.overlap = overlap
        self.hop_length = int(frame_size * (1 - overlap))
        
        # Speed of sound (m/s)
        self.c = 343.0
        
        # Generate mic pairs for GCC-PHAT
        self.mic_pairs = [(i, j) for i in range(4) for j in range(i+1, 4)]
        
        print(f"Initialized ZODAS Pipeline:")
        print(f"  - Sample Rate: {sample_rate} Hz")
        print(f"  - Frame Size: {frame_size}")
        print(f"  - Mic Pairs: {self.mic_pairs}")
    
    def load_multichannel_audio(self, wav_files: List[str]) -> np.ndarray:
        """
        Load separate mono WAV files and combine into multichannel array
        
        Args:
            wav_files: List of paths to mono WAV files (one per mic)
            
        Returns:
            audio_data: Shape (n_samples, n_channels)
        """
        channels = []
        sample_rates = []
        
        for i, wav_file in enumerate(wav_files):
            try:
                sr, data = wavfile.read(wav_file)
                
                # Convert to float32 and normalize
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                
                channels.append(data)
                sample_rates.append(sr)
                
            except Exception as e:
                print(f"Error loading {wav_file}: {e}")
                raise
        
        # Verify all files have same sample rate and length
        if len(set(sample_rates)) > 1:
            raise ValueError(f"Sample rates don't match: {sample_rates}")
        
        if len(set(len(ch) for ch in channels)) > 1:
            min_len = min(len(ch) for ch in channels)
            channels = [ch[:min_len] for ch in channels]
            print(f"Warning: Trimmed to {min_len} samples")
        
        return np.column_stack(channels)
    
    def apply_windowing(self, frame: np.ndarray) -> np.ndarray:
        """Apply Hann window to frame"""
        window = scipy.signal.windows.hann(self.frame_size)
        return frame * window[:, np.newaxis]
    
    def compute_stft(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform for all channels
        
        Returns:
            stft_data: Complex STFT coefficients (n_frames, n_freq_bins, n_channels)
            freqs: Frequency bins
        """
        n_samples, n_channels = audio_data.shape
        n_frames = (n_samples - self.frame_size) // self.hop_length + 1
        n_freq_bins = self.frame_size // 2 + 1
        
        stft_data = np.zeros((n_frames, n_freq_bins, n_channels), dtype=np.complex64)
        
        for frame_idx in range(n_frames):
            start = frame_idx * self.hop_length
            end = start + self.frame_size
            
            frame = audio_data[start:end, :]
            windowed_frame = self.apply_windowing(frame)
            
            # Compute FFT for each channel
            for ch in range(n_channels):
                fft_result = scipy.fft.rfft(windowed_frame[:, ch])
                stft_data[frame_idx, :, ch] = fft_result
        
        freqs = scipy.fft.rfftfreq(self.frame_size, 1/self.sample_rate)
        return stft_data, freqs
    
    def gcc_phat(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Generalized Cross-Correlation with Phase Transform
        
        Args:
            X1, X2: FFT coefficients for two microphones
            
        Returns:
            gcc: Cross-correlation function
            tdoa: Time difference of arrival (in samples)
        """
        # Cross-spectrum
        cross_spectrum = X1 * np.conj(X2)
        
        # Phase transform (avoid division by zero)
        magnitude = np.abs(cross_spectrum)
        magnitude[magnitude < 1e-10] = 1e-10
        phat = cross_spectrum / magnitude
        
        # Inverse FFT to get cross-correlation
        gcc = scipy.fft.irfft(phat, n=self.frame_size)
        
        # Find peak (TDOA in samples)
        max_idx = np.argmax(np.abs(gcc))
        if max_idx > self.frame_size // 2:
            max_idx -= self.frame_size
        
        return gcc, max_idx
    
    def tdoa_to_doa(self, tdoa_samples: Dict[Tuple[int, int], float]) -> Tuple[float, float]:
        """
        Convert TDOA measurements to Direction of Arrival (DOA)
        Using least squares estimation for 2D DOA (azimuth, elevation)
        
        Returns:
            azimuth: Azimuth angle in degrees (-180 to 180)
            elevation: Elevation angle in degrees (-90 to 90)
        """
        # Build system of equations: tdoa = (r_j - r_i) · s / c
        A = []
        b = []
        
        for (i, j), tdoa in tdoa_samples.items():
            # Direction vector difference
            diff = self.mic_positions[j] - self.mic_positions[i]
            A.append(diff)
            b.append(tdoa * self.c)
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve for unit direction vector using least squares
        try:
            s, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
            
            # Normalize to unit vector
            s_norm = np.linalg.norm(s)
            if s_norm > 1e-10:
                s = s / s_norm
            
            # Convert to spherical coordinates
            x, y, z = s
            azimuth = np.degrees(np.arctan2(y, x))
            elevation = np.degrees(np.arcsin(np.clip(z, -1, 1)))
            
            return azimuth, elevation
            
        except np.linalg.LinAlgError:
            return 0.0, 0.0
    
    def find_spectral_peaks(self, magnitude_spectrum: np.ndarray, n_peaks: int = 10) -> List[Tuple[int, float]]:
        """
        Find top N spectral peaks across all channels
        
        Returns:
            List of (frequency_bin, magnitude) tuples
        """
        # Average magnitude across channels
        avg_magnitude = np.mean(magnitude_spectrum, axis=1)
        
        # Find peaks
        peaks, properties = scipy.signal.find_peaks(avg_magnitude, height=0.01 * np.max(avg_magnitude))
        
        # Sort by magnitude and take top N
        peak_magnitudes = avg_magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        top_peaks = []
        for i in sorted_indices[:n_peaks]:
            peak_idx = peaks[i]
            top_peaks.append((peak_idx, peak_magnitudes[i]))
        
        return top_peaks
    
    def process_frame(self, stft_frame: np.ndarray, freqs: np.ndarray) -> Dict:
        """
        Process a single STFT frame to extract DOA and spectral features
        
        Args:
            stft_frame: Shape (n_freq_bins, n_channels)
            freqs: Frequency bins
            
        Returns:
            Dictionary with DOA estimates and spectral features
        """
        results = {
            'tdoa_estimates': {},
            'doa_azimuth': 0.0,
            'doa_elevation': 0.0,
            'spectral_peaks': [],
            'magnitude_spectrum': np.abs(stft_frame)
        }
        
        # Compute TDOA for each mic pair
        for i, j in self.mic_pairs:
            gcc, tdoa = self.gcc_phat(stft_frame[:, i], stft_frame[:, j])
            tdoa_seconds = tdoa / self.sample_rate
            results['tdoa_estimates'][(i, j)] = tdoa_seconds
        
        # Estimate DOA from TDOA measurements
        azimuth, elevation = self.tdoa_to_doa(results['tdoa_estimates'])
        results['doa_azimuth'] = azimuth
        results['doa_elevation'] = elevation
        
        # Find spectral peaks
        peaks = self.find_spectral_peaks(results['magnitude_spectrum'])
        results['spectral_peaks'] = [(freqs[bin_idx], mag) for bin_idx, mag in peaks]
        
        return results
    
    def create_spatial_spectrogram(self, stft_data: np.ndarray, frame_indices: List[int]) -> np.ndarray:
        """
        Create Spatial Spectrogram (SS) matrix for classifier input
        
        Args:
            stft_data: Full STFT data (n_frames, n_freq_bins, n_channels)
            frame_indices: List of frame indices to include in SS matrix
            
        Returns:
            SS matrix: Shape (n_selected_frames, n_freq_bins)
        """
        ss_matrix = []
        
        for frame_idx in frame_indices:
            if frame_idx < stft_data.shape[0]:
                # Average magnitude across all mic pairs (spectral product approach)
                frame_data = stft_data[frame_idx]
                
                spectral_products = []
                for i, j in self.mic_pairs:
                    product = np.abs(frame_data[:, i]) * np.abs(frame_data[:, j])
                    spectral_products.append(product)
                
                # Average across mic pairs
                avg_spectrum = np.mean(spectral_products, axis=0)
                ss_matrix.append(avg_spectrum)
        
        return np.array(ss_matrix)
    
    def process_audio_files(self, wav_files: List[str], ground_truth_positions: Optional[List] = None) -> Dict:
        """
        Main processing function for the ZODAS pipeline
        
        Args:
            wav_files: List of 4 mono WAV files (one per mic)
            ground_truth_positions: Optional list of known source positions for validation
                                  Format: [{'position': [x, y, z], 'azimuth': az, 'elevation': el}, ...]
            
        Returns:
            Complete analysis results including validation metrics if ground truth provided
        """
        print("Loading audio files...")
        audio_data = self.load_multichannel_audio(wav_files)
        
        print("Computing STFT...")
        stft_data, freqs = self.compute_stft(audio_data)
        n_frames = stft_data.shape[0]
        
        print(f"Processing {n_frames} frames...")
        frame_results = []
        
        for frame_idx in range(n_frames):
            frame_result = self.process_frame(stft_data[frame_idx], freqs)
            frame_result['frame_index'] = frame_idx
            frame_result['timestamp'] = frame_idx * self.hop_length / self.sample_rate
            frame_results.append(frame_result)
        
        # Create spatial spectrogram for potential elephant frames
        # (In practice, you'd select frames based on detection criteria)
        selected_frames = list(range(0, min(8, n_frames)))  # Take first 8 frames as example
        ss_matrix = self.create_spatial_spectrogram(stft_data, selected_frames)
        
        results = {
            'frame_results': frame_results,
            'spatial_spectrogram': ss_matrix,
            'freqs': freqs,
            'stft_data': stft_data,
            'audio_duration': len(audio_data) / self.sample_rate,
            'n_frames_processed': n_frames
        }
        
        # Add validation metrics if ground truth is provided
        if ground_truth_positions:
            validation_metrics = self.compute_validation_metrics(frame_results, ground_truth_positions)
            results['validation_metrics'] = validation_metrics
        
        return results
    
    def compute_validation_metrics(self, frame_results: List[Dict], ground_truth_positions: List[Dict]) -> Dict:
        """
        Compute validation metrics comparing ZODAS estimates to ground truth
        
        Args:
            frame_results: List of frame processing results from ZODAS
            ground_truth_positions: List of ground truth source positions
            
        Returns:
            Dictionary containing various validation metrics
        """
        # Extract DOA estimates over time
        estimated_azimuths = [r['doa_azimuth'] for r in frame_results]
        estimated_elevations = [r['doa_elevation'] for r in frame_results]
        timestamps = [r['timestamp'] for r in frame_results]
        
        # For now, assume single source scenario (most common for elephant detection)
        # In multi-source scenarios, you'd need source association/tracking
        gt_source = ground_truth_positions[0] if ground_truth_positions else None
        
        if not gt_source:
            return {'error': 'No ground truth provided'}
        
        gt_azimuth = gt_source.get('azimuth', 0.0)
        gt_elevation = gt_source.get('elevation', 0.0)
        
        # Compute error metrics
        azimuth_errors = []
        elevation_errors = []
        
        for est_az, est_el in zip(estimated_azimuths, estimated_elevations):
            # Angular error calculation (handling wrap-around)
            az_error = self._angular_error(est_az, gt_azimuth)
            el_error = abs(est_el - gt_elevation)
            
            azimuth_errors.append(az_error)
            elevation_errors.append(el_error)
        
        # Statistical metrics
        metrics = {
            'ground_truth': {
                'azimuth': gt_azimuth,
                'elevation': gt_elevation,
                'position': gt_source.get('position', [0, 0, 0])
            },
            'estimates': {
                'azimuths': estimated_azimuths,
                'elevations': estimated_elevations,
                'timestamps': timestamps
            },
            'errors': {
                'azimuth_errors': azimuth_errors,
                'elevation_errors': elevation_errors,
                'mean_azimuth_error': np.mean(azimuth_errors),
                'std_azimuth_error': np.std(azimuth_errors),
                'mean_elevation_error': np.mean(elevation_errors),
                'std_elevation_error': np.std(elevation_errors),
                'median_azimuth_error': np.median(azimuth_errors),
                'median_elevation_error': np.median(elevation_errors),
                'max_azimuth_error': np.max(azimuth_errors),
                'max_elevation_error': np.max(elevation_errors)
            },
            'accuracy_metrics': {
                'azimuth_accuracy_5deg': np.mean(np.array(azimuth_errors) <= 5.0) * 100,
                'azimuth_accuracy_10deg': np.mean(np.array(azimuth_errors) <= 10.0) * 100,
                'azimuth_accuracy_15deg': np.mean(np.array(azimuth_errors) <= 15.0) * 100,
                'elevation_accuracy_5deg': np.mean(np.array(elevation_errors) <= 5.0) * 100,
                'elevation_accuracy_10deg': np.mean(np.array(elevation_errors) <= 10.0) * 100,
            },
            'stability_metrics': {
                'azimuth_variance': np.var(estimated_azimuths),
                'elevation_variance': np.var(estimated_elevations),
                'temporal_consistency': self._compute_temporal_consistency(estimated_azimuths, estimated_elevations)
            }
        }
        
        # Add confidence intervals
        confidence_level = 0.95
        metrics['confidence_intervals'] = self._compute_confidence_intervals(
            azimuth_errors, elevation_errors, confidence_level
        )
        
        return metrics
    
    def _angular_error(self, estimated: float, ground_truth: float) -> float:
        """Compute angular error between two angles in degrees, handling wrap-around"""
        error = abs(estimated - ground_truth)
        if error > 180:
            error = 360 - error
        return error
    
    def _compute_temporal_consistency(self, azimuths: List[float], elevations: List[float]) -> float:
        """Compute temporal consistency metric (lower is better)"""
        if len(azimuths) < 2:
            return 0.0
        
        # Compute frame-to-frame variations
        az_diffs = [self._angular_error(azimuths[i], azimuths[i-1]) for i in range(1, len(azimuths))]
        el_diffs = [abs(elevations[i] - elevations[i-1]) for i in range(1, len(elevations))]
        
        # Average temporal variation
        return (np.mean(az_diffs) + np.mean(el_diffs)) / 2
    
    def _compute_confidence_intervals(self, azimuth_errors: List[float], elevation_errors: List[float], 
                                   confidence_level: float = 0.95) -> Dict:
        """Compute confidence intervals for error metrics"""
        from scipy import stats
        
        alpha = 1 - confidence_level
        
        # For azimuth errors
        az_mean = np.mean(azimuth_errors)
        az_std = np.std(azimuth_errors, ddof=1)
        az_n = len(azimuth_errors)
        az_sem = az_std / np.sqrt(az_n)
        az_ci = stats.t.interval(confidence_level, az_n-1, loc=az_mean, scale=az_sem)
        
        # For elevation errors
        el_mean = np.mean(elevation_errors)
        el_std = np.std(elevation_errors, ddof=1)
        el_n = len(elevation_errors)
        el_sem = el_std / np.sqrt(el_n)
        el_ci = stats.t.interval(confidence_level, el_n-1, loc=el_mean, scale=el_sem)
        
        return {
            'azimuth_ci_lower': az_ci[0],
            'azimuth_ci_upper': az_ci[1],
            'elevation_ci_lower': el_ci[0],
            'elevation_ci_upper': el_ci[1],
            'confidence_level': confidence_level
        }
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create comprehensive visualization of the processing results including validation metrics
        """
        # Determine if we have validation data
        has_validation = 'validation_metrics' in results
        
        if has_validation:
            fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        else:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # DOA trajectory
        azimuths = [r['doa_azimuth'] for r in results['frame_results']]
        elevations = [r['doa_elevation'] for r in results['frame_results']]
        timestamps = [r['timestamp'] for r in results['frame_results']]
        
        axes[0, 0].plot(timestamps, azimuths, 'b-', label='Estimated', linewidth=2)
        if has_validation:
            gt_az = results['validation_metrics']['ground_truth']['azimuth']
            axes[0, 0].axhline(y=gt_az, color='red', linestyle='--', label='Ground Truth', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Azimuth (degrees)')
        axes[0, 0].set_title('DOA Azimuth Over Time')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        axes[0, 1].plot(timestamps, elevations, 'g-', label='Estimated', linewidth=2)
        if has_validation:
            gt_el = results['validation_metrics']['ground_truth']['elevation']
            axes[0, 1].axhline(y=gt_el, color='red', linestyle='--', label='Ground Truth', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Elevation (degrees)')
        axes[0, 1].set_title('DOA Elevation Over Time')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # 2D DOA scatter plot
        scatter = axes[0, 2].scatter(azimuths, elevations, c=timestamps, cmap='viridis', alpha=0.7, s=30)
        if has_validation:
            gt_az = results['validation_metrics']['ground_truth']['azimuth']
            gt_el = results['validation_metrics']['ground_truth']['elevation']
            axes[0, 2].scatter(gt_az, gt_el, c='red', marker='*', s=200, label='Ground Truth')
            axes[0, 2].legend()
        axes[0, 2].set_xlabel('Azimuth (degrees)')
        axes[0, 2].set_ylabel('Elevation (degrees)')
        axes[0, 2].set_title('DOA Trajectory (2D)')
        axes[0, 2].grid(True)
        plt.colorbar(scatter, ax=axes[0, 2], label='Time (s)')
        
        # Spatial Spectrogram
        if results['spatial_spectrogram'].size > 0:
            im = axes[1, 0].imshow(results['spatial_spectrogram'].T, 
                                 aspect='auto', origin='lower', cmap='viridis')
            axes[1, 0].set_xlabel('Frame Index')
            axes[1, 0].set_ylabel('Frequency Bin')
            axes[1, 0].set_title('Spatial Spectrogram')
            plt.colorbar(im, ax=axes[1, 0])
        
        # Average magnitude spectrum
        avg_spectrum = np.mean([np.mean(r['magnitude_spectrum'], axis=1) 
                               for r in results['frame_results']], axis=0)
        axes[1, 1].plot(results['freqs'], avg_spectrum)
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].set_title('Average Magnitude Spectrum')
        axes[1, 1].grid(True)
        
        # Spectral peaks over time
        peak_freqs = []
        peak_times = []
        for r in results['frame_results']:
            for freq, mag in r['spectral_peaks'][:5]:  # Top 5 peaks
                peak_freqs.append(freq)
                peak_times.append(r['timestamp'])
        
        axes[1, 2].scatter(peak_times, peak_freqs, alpha=0.6, s=20)
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Frequency (Hz)')
        axes[1, 2].set_title('Spectral Peaks Over Time')
        axes[1, 2].grid(True)
        
        # Validation plots (if available)
        if has_validation:
            vm = results['validation_metrics']
            
            # Error over time
            ax_err = axes[2, 0]
            ax_err.plot(timestamps, vm['errors']['azimuth_errors'], 'b-', label='Azimuth Error', linewidth=2)
            ax_err.plot(timestamps, vm['errors']['elevation_errors'], 'g-', label='Elevation Error', linewidth=2)
            ax_err.set_xlabel('Time (s)')
            ax_err.set_ylabel('Error (degrees)')
            ax_err.set_title('Estimation Errors Over Time')
            ax_err.legend()
            ax_err.grid(True)
            
            # Error statistics
            ax_stats = axes[2, 1]
            error_types = ['Mean Az', 'Std Az', 'Mean El', 'Std El', 'Max Az', 'Max El']
            error_values = [
                vm['errors']['mean_azimuth_error'],
                vm['errors']['std_azimuth_error'],
                vm['errors']['mean_elevation_error'],
                vm['errors']['std_elevation_error'],
                vm['errors']['max_azimuth_error'],
                vm['errors']['max_elevation_error']
            ]
            colors = ['blue', 'lightblue', 'green', 'lightgreen', 'red', 'lightcoral']
            bars = ax_stats.bar(error_types, error_values, color=colors)
            ax_stats.set_ylabel('Error (degrees)')
            ax_stats.set_title('Error Statistics')
            ax_stats.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, error_values):
                height = bar.get_height()
                ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                             f'{value:.1f}°', ha='center', va='bottom', fontsize=9)
            
            # Accuracy metrics
            ax_acc = axes[2, 2]
            accuracy_labels = ['5°', '10°', '15°']
            az_acc = [vm['accuracy_metrics']['azimuth_accuracy_5deg'],
                     vm['accuracy_metrics']['azimuth_accuracy_10deg'],
                     vm['accuracy_metrics']['azimuth_accuracy_15deg']]
            el_acc = [vm['accuracy_metrics']['elevation_accuracy_5deg'],
                     vm['accuracy_metrics']['elevation_accuracy_10deg'],
                     0]  # 15° not computed for elevation
            
            x = np.arange(len(accuracy_labels))
            width = 0.35
            
            bars1 = ax_acc.bar(x - width/2, az_acc, width, label='Azimuth', color='blue', alpha=0.7)
            bars2 = ax_acc.bar(x + width/2, el_acc[:2] + [0], width, label='Elevation', color='green', alpha=0.7)
            
            ax_acc.set_ylabel('Accuracy (%)')
            ax_acc.set_title('Accuracy within Tolerance')
            ax_acc.set_xticks(x)
            ax_acc.set_xticklabels(accuracy_labels)
            ax_acc.legend()
            ax_acc.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax_acc.text(bar.get_x() + bar.get_width()/2., height + 1,
                                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_validation_report(self, results: Dict) -> str:
        """Generate a text report of validation metrics"""
        if 'validation_metrics' not in results:
            return "No validation metrics available."
        
        vm = results['validation_metrics']
        
        report = f"""
ZODAS Validation Report
======================

Ground Truth:
- Azimuth: {vm['ground_truth']['azimuth']:.1f}°
- Elevation: {vm['ground_truth']['elevation']:.1f}°

Error Statistics:
- Mean Azimuth Error: {vm['errors']['mean_azimuth_error']:.2f}° ± {vm['errors']['std_azimuth_error']:.2f}°
- Mean Elevation Error: {vm['errors']['mean_elevation_error']:.2f}° ± {vm['errors']['std_elevation_error']:.2f}°
- Median Azimuth Error: {vm['errors']['median_azimuth_error']:.2f}°
- Median Elevation Error: {vm['errors']['median_elevation_error']:.2f}°
- Maximum Azimuth Error: {vm['errors']['max_azimuth_error']:.2f}°
- Maximum Elevation Error: {vm['errors']['max_elevation_error']:.2f}°

Accuracy Metrics:
- Azimuth accuracy within 5°: {vm['accuracy_metrics']['azimuth_accuracy_5deg']:.1f}%
- Azimuth accuracy within 10°: {vm['accuracy_metrics']['azimuth_accuracy_10deg']:.1f}%
- Azimuth accuracy within 15°: {vm['accuracy_metrics']['azimuth_accuracy_15deg']:.1f}%
- Elevation accuracy within 5°: {vm['accuracy_metrics']['elevation_accuracy_5deg']:.1f}%
- Elevation accuracy within 10°: {vm['accuracy_metrics']['elevation_accuracy_10deg']:.1f}%

Stability Metrics:
- Azimuth variance: {vm['stability_metrics']['azimuth_variance']:.2f}
- Elevation variance: {vm['stability_metrics']['elevation_variance']:.2f}
- Temporal consistency: {vm['stability_metrics']['temporal_consistency']:.2f}° (lower is better)

Confidence Intervals (95%):
- Azimuth error: [{vm['confidence_intervals']['azimuth_ci_lower']:.2f}°, {vm['confidence_intervals']['azimuth_ci_upper']:.2f}°]
- Elevation error: [{vm['confidence_intervals']['elevation_ci_lower']:.2f}°, {vm['confidence_intervals']['elevation_ci_upper']:.2f}°]

Total frames processed: {len(vm['estimates']['timestamps'])}
Processing duration: {vm['estimates']['timestamps'][-1]:.2f}s
"""
        
        return report

# Example usage function
def run_zodas_analysis(wav_files: List[str], ground_truth: Optional[Dict] = None):
    """
    Example function to run the complete ZODAS analysis
    
    Args:
        wav_files: List of 4 mono WAV files [mic1.wav, mic2.wav, mic3.wav, mic4.wav]
        ground_truth: Optional dict with known elephant positions
    """
    # Initialize pipeline
    zodas = ZODASPipeline(sample_rate=16000, frame_size=512, overlap=0.5)
    
    # Process audio
    results = zodas.process_audio_files(wav_files, ground_truth)
    
    # Print summary
    print(f"\n=== ZODAS Analysis Results ===")
    print(f"Audio Duration: {results['audio_duration']:.2f} seconds")
    print(f"Frames Processed: {results['n_frames_processed']}")
    print(f"Spatial Spectrogram Shape: {results['spatial_spectrogram'].shape}")
    
    # Show some DOA estimates
    print(f"\nDOA Estimates (first 5 frames):")
    for i in range(min(5, len(results['frame_results']))):
        r = results['frame_results'][i]
        print(f"  Frame {i}: Az={r['doa_azimuth']:.1f}°, El={r['doa_elevation']:.1f}°")
    
    # Visualize results
    zodas.visualize_results(results)
    
    return results

# Example usage:
# wav_files = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']
# results = run_zodas_analysis(wav_files)